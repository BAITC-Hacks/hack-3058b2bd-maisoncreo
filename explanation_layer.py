"""Optional, grounded explanations for already-selected contractors."""

from copy import deepcopy
import json
import os
import re
from urllib import request


DEFAULT_MODEL = "gpt-5.4-mini"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

STRUCTURED_FIELD_ORDER = (
    "category",
    "city",
    "event_format",
    "price_from_kzt",
    "language",
    "max_hours",
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "description_evidence": {
            "type": "string",
            "description": (
                "A verbatim substring copied from contractor.description. Do not add "
                "quotation marks, punctuation, prefixes, suffixes, ellipses, "
                "normalization, paraphrasing, or formatting."
            ),
        },
        "structured_evidence": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": list(STRUCTURED_FIELD_ORDER),
            },
        },
    },
    "required": [
        "description_evidence",
        "structured_evidence",
    ],
    "additionalProperties": False,
}


class ExplanationValidationError(ValueError):
    """Raised when a generated explanation is not sufficiently grounded."""


def enrich_recommendations(result, records, query, *, provider=None):
    """Add grounded explanations without changing selection or ordering.

    A provider is a callable accepting ``(contractor, query)`` and returning the
    structured object described by ``OUTPUT_SCHEMA``. When omitted, OpenAI is
    used only if ``OPENAI_API_KEY`` is present. Every failure uses the local
    deterministic fallback.
    """
    enriched = deepcopy(result)
    records_by_id = {str(record["id"]): record for record in records}
    selected_ids = [str(card["id"]) for card in result.get("recommendations", [])]

    if provider is None and os.getenv("OPENAI_API_KEY"):
        provider = openai_explanation

    provider_available = provider is not None
    for card in enriched.get("recommendations", []):
        contractor = records_by_id.get(str(card["id"]))
        if contractor is None:
            continue

        generated = None
        if provider_available:
            try:
                generated = validate_explanation(provider(contractor, query), contractor, query)
            except Exception:
                generated = None
                provider_available = False

        if generated is None:
            generated = fallback_explanation(contractor, query)
            source = "fallback"
        else:
            source = "openai"

        card["explanation"] = generated["explanation"]
        card["explanation_evidence"] = {
            "structured_fields": generated["structured_evidence"],
            "description_excerpt": generated["description_evidence"],
        }
        card["explanation_source"] = source

    assert [str(card["id"]) for card in enriched.get("recommendations", [])] == selected_ids
    return enriched


def openai_explanation(contractor, query):
    """Request one structured explanation from the OpenAI Responses API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    model = os.getenv("OPENAI_EXPLANATION_MODEL", DEFAULT_MODEL)
    safe_input = {
        "contractor": {
            "anon_name": contractor.get("anon_name"),
            "categories": contractor.get("categories"),
            "city": contractor.get("city"),
            "price_from_kzt": contractor.get("price_from_kzt"),
            "event_formats": contractor.get("event_formats"),
            "languages": contractor.get("languages"),
            "max_hours": contractor.get("max_hours"),
            "description": contractor.get("description", ""),
        },
        "verified_request": {
            "category": query.get("category"),
            "city": query.get("city"),
            "event_format": query.get("event_format"),
            "budget_kzt": query.get("budget_kzt"),
            "language": query.get("language"),
            "duration_hours": query.get("duration_hours"),
        },
    }
    body = {
        "model": model,
        "store": False,
        "max_output_tokens": 300,
        "input": [
            {
                "role": "system",
                "content": (
                    "Select evidence for a concise Russian contractor explanation. "
                    "Return at least one structured_evidence field that matches the "
                    "verified request and contractor data. "
                    "Copy description_evidence verbatim from contractor.description: it "
                    "must be an exact source substring. Do not add quotation marks, "
                    "punctuation, prefixes, suffixes, ellipses, normalization, "
                    "paraphrasing, or formatting to description_evidence. Do not write "
                    "the final explanation; the application constructs it deterministically."
                ),
            },
            {"role": "user", "content": json.dumps(safe_input, ensure_ascii=False)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "contractor_explanation",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
    }

    http_request = request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(http_request, timeout=6) as response:
        api_response = json.loads(response.read().decode("utf-8"))
    return json.loads(_extract_output_text(api_response))


def validate_explanation(output, contractor, query):
    """Return validated model output or raise ``ExplanationValidationError``."""
    if not isinstance(output, dict) or set(output) != {
        "description_evidence",
        "structured_evidence",
    }:
        raise ExplanationValidationError("Unexpected explanation shape")

    description_evidence = output["description_evidence"]
    structured_evidence = output["structured_evidence"]
    if not isinstance(description_evidence, str) or not description_evidence.strip():
        raise ExplanationValidationError("Description evidence must be a non-empty string")
    if not isinstance(structured_evidence, list) or not structured_evidence:
        raise ExplanationValidationError("At least one structured evidence field is required")
    if not all(isinstance(field, str) for field in structured_evidence):
        raise ExplanationValidationError("Structured evidence fields must be strings")
    if len(structured_evidence) != len(set(structured_evidence)):
        raise ExplanationValidationError("Structured evidence fields must be unique")

    description = str(contractor.get("description", ""))
    if description_evidence not in description:
        raise ExplanationValidationError("Description evidence is not an exact source excerpt")

    allowed_fields = set(OUTPUT_SCHEMA["properties"]["structured_evidence"]["items"]["enum"])
    if any(field not in allowed_fields for field in structured_evidence):
        raise ExplanationValidationError("Unknown structured evidence field")
    for field in structured_evidence:
        if not _structured_field_is_verified(field, contractor, query):
            raise ExplanationValidationError(f"Structured field {field!r} is not grounded")

    ordered_evidence = [
        field for field in STRUCTURED_FIELD_ORDER if field in structured_evidence
    ]
    explanation = _build_explanation(
        contractor, query, ordered_evidence, description_evidence
    )
    if len(explanation) > 600 or not 1 <= _sentence_count(explanation) <= 2:
        raise ExplanationValidationError("Generated explanation must contain one or two sentences")

    return {
        "explanation": explanation,
        "description_evidence": description_evidence,
        "structured_evidence": ordered_evidence,
    }


def fallback_explanation(contractor, query):
    """Build a deterministic grounded explanation without an API call."""
    excerpt = _description_excerpt(str(contractor.get("description", "")))
    explanation = _build_explanation(
        contractor, query, ["event_format", "price_from_kzt"], excerpt
    )
    return {
        "explanation": explanation,
        "description_evidence": excerpt,
        "structured_evidence": ["event_format", "price_from_kzt"],
    }


def _structured_field_is_verified(field, contractor, query):
    if field == "category":
        return _contains_token(contractor.get("categories"), query.get("category"))
    if field == "city":
        return _same_text(contractor.get("city"), query.get("city"))
    if field == "event_format":
        return _contains_token(contractor.get("event_formats"), query.get("event_format"))
    if field == "language":
        return bool(query.get("language")) and _contains_token(
            contractor.get("languages"), query.get("language")
        )
    if field == "max_hours":
        try:
            return float(contractor["max_hours"]) >= float(query["duration_hours"])
        except (KeyError, TypeError, ValueError):
            return False
    if field == "price_from_kzt":
        try:
            return int(contractor["price_from_kzt"]) <= int(query["budget_kzt"])
        except (KeyError, TypeError, ValueError):
            return False
    return False


def _build_explanation(contractor, query, structured_evidence, description_evidence):
    facts = []
    for field in structured_evidence:
        if field == "category":
            facts.append(f"категория «{query['category']}»")
        elif field == "city":
            facts.append(f"город — {query['city']}")
        elif field == "event_format":
            facts.append(f"подходит для формата «{query['event_format']}»")
        elif field == "price_from_kzt":
            price = f"{int(contractor['price_from_kzt']):,}".replace(",", " ")
            facts.append(f"стартовая цена — от {price} KZT")
        elif field == "language":
            facts.append(f"заявлен язык «{query['language']}»")
        elif field == "max_hours":
            facts.append(
                f"лимит {contractor['max_hours']} ч покрывает запрос "
                f"на {query['duration_hours']} ч"
            )

    name = str(contractor.get("anon_name", "Подрядчик")).strip() or "Подрядчик"
    explanation = f"{name}: {'; '.join(facts)}."
    if description_evidence:
        explanation += f" В описании указано: «{description_evidence}»."
    return explanation


def _contains_token(value, expected):
    expected_text = str(expected or "").strip().casefold()
    return bool(expected_text) and expected_text in {
        token.strip().casefold() for token in str(value or "").split("|")
    }


def _same_text(left, right):
    return bool(str(right or "").strip()) and str(left or "").strip().casefold() == str(
        right
    ).strip().casefold()


def _description_excerpt(description):
    candidates = [
        match.group().strip().rstrip(".!?")
        for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", description)
    ]
    candidates = [part for part in candidates if part]
    preferred = [part for part in candidates if len(part.split()) >= 4]
    for candidate in preferred:
        intro = candidate.casefold()
        if not intro.startswith(("я ", "я,", "меня зовут", "привет")) and " — " not in candidate[:60]:
            return _exact_word_prefix(candidate, 24)
    if preferred:
        return _exact_word_prefix(preferred[0], 24)
    if candidates:
        return _exact_word_prefix(candidates[0], 24)
    return ""


def _exact_word_prefix(text, word_limit):
    words = list(re.finditer(r"\S+", text))
    if len(words) <= word_limit:
        return text
    return text[: words[word_limit - 1].end()]


def _sentence_count(text):
    return len([part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part])


def _extract_output_text(api_response):
    for output in api_response.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"]
    raise ValueError("OpenAI response did not contain output text")
