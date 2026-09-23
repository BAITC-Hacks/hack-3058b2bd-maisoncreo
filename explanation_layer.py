"""Optional, grounded explanations for already-selected contractors."""

from copy import deepcopy
import json
import os
import re
from urllib import request


DEFAULT_MODEL = "gpt-5.4-mini"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

GENERIC_PHRASES = (
    "excellent choice",
    "great choice",
    "perfect choice",
    "отличный выбор",
    "идеальный выбор",
    "идеально подойдет",
    "идеально подойдёт",
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
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
                "enum": [
                    "category",
                    "city",
                    "event_format",
                    "price_from_kzt",
                    "language",
                    "max_hours",
                ],
            },
        },
    },
    "required": [
        "explanation",
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

    for card in enriched.get("recommendations", []):
        contractor = records_by_id.get(str(card["id"]))
        if contractor is None:
            continue

        generated = None
        if provider is not None:
            try:
                generated = validate_explanation(provider(contractor, query), contractor, query)
            except Exception:
                generated = None

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
                    "Write a concise Russian contractor explanation in 1-2 sentences. "
                    "Use only the supplied verified fields and description. Start with "
                    "anon_name and never use another person or company name as the "
                    "contractor name. Mention at least one verified structured match. "
                    "Copy description_evidence verbatim from contractor.description: it "
                    "must be an exact source substring. Do not add quotation marks, "
                    "punctuation, prefixes, suffixes, ellipses, normalization, "
                    "paraphrasing, or formatting to description_evidence. The explanation "
                    "may use natural Russian normally, but must include the exact "
                    "description_evidence substring. Never discuss "
                    "availability, ratings, reviews, capacity, or experience unless stated "
                    "verbatim in the evidence excerpt. If mentioning price, call it a "
                    "starting price using «от» or «стартовая цена». Avoid generic praise."
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
    with request.urlopen(http_request, timeout=20) as response:
        api_response = json.loads(response.read().decode("utf-8"))
    return json.loads(_extract_output_text(api_response))


def validate_explanation(output, contractor, query):
    """Return validated model output or raise ``ExplanationValidationError``."""
    if not isinstance(output, dict) or set(output) != {
        "explanation",
        "description_evidence",
        "structured_evidence",
    }:
        raise ExplanationValidationError("Unexpected explanation shape")

    explanation = output["explanation"]
    description_evidence = output["description_evidence"]
    structured_evidence = output["structured_evidence"]
    if not all(isinstance(value, str) and value.strip() for value in (explanation, description_evidence)):
        raise ExplanationValidationError("Explanation and evidence must be non-empty strings")
    if not isinstance(structured_evidence, list) or not structured_evidence:
        raise ExplanationValidationError("At least one structured evidence field is required")
    if len(explanation) > 600 or not 1 <= _sentence_count(explanation) <= 2:
        raise ExplanationValidationError("Explanation must contain one or two sentences")

    display_name = str(contractor.get("anon_name", "")).strip()
    if not display_name or not explanation.startswith(display_name):
        raise ExplanationValidationError("Explanation must start with anon_name")

    description = str(contractor.get("description", ""))
    if description_evidence not in description:
        raise ExplanationValidationError("Description evidence is not an exact source excerpt")
    if description_evidence.casefold() not in explanation.casefold():
        raise ExplanationValidationError("Explanation does not contain its description evidence")

    lowered = explanation.casefold()
    if any(phrase in lowered for phrase in GENERIC_PHRASES):
        raise ExplanationValidationError("Generic praise is not allowed")
    if "доступ" in lowered or "свобод" in lowered or query.get("event_date", "") in explanation:
        raise ExplanationValidationError("The explanation must not infer availability")
    if any(marker in lowered for marker in ("kzt", "₸", "тенге")) and not (
        "старт" in lowered or re.search(r"(?:^|\s)от(?:\s|$)", lowered)
    ):
        raise ExplanationValidationError("Price must be described as a starting price")

    allowed_fields = set(OUTPUT_SCHEMA["properties"]["structured_evidence"]["items"]["enum"])
    if any(field not in allowed_fields for field in structured_evidence):
        raise ExplanationValidationError("Unknown structured evidence field")
    for field in structured_evidence:
        if not _structured_field_is_mentioned(field, explanation, contractor, query):
            raise ExplanationValidationError(f"Structured field {field!r} is not grounded")

    return {
        "explanation": explanation.strip(),
        "description_evidence": description_evidence.strip(),
        "structured_evidence": list(structured_evidence),
    }


def fallback_explanation(contractor, query):
    """Build a deterministic grounded explanation without an API call."""
    name = str(contractor.get("anon_name", "Подрядчик"))
    price = int(contractor["price_from_kzt"])
    excerpt = _description_excerpt(str(contractor.get("description", "")))
    explanation = (
        f"{name}: формат «{query['event_format']}», стартовая цена — от "
        f"{price:,} KZT. В описании указано: «{excerpt}»."
    )
    return {
        "explanation": explanation,
        "description_evidence": excerpt,
        "structured_evidence": ["event_format", "price_from_kzt"],
    }


def _structured_field_is_mentioned(field, explanation, contractor, query):
    lowered = explanation.casefold()
    if field == "category":
        return str(query.get("category", "")).casefold() in lowered
    if field == "city":
        return str(query.get("city", "")).casefold() in lowered
    if field == "event_format":
        return str(query.get("event_format", "")).casefold() in lowered
    if field == "language":
        language = query.get("language")
        return bool(language) and str(language).casefold() in lowered
    if field == "max_hours":
        hours = contractor.get("max_hours")
        return hours not in (None, "") and str(hours) in explanation
    if field == "price_from_kzt":
        price = str(int(contractor["price_from_kzt"]))
        return price in re.sub(r"\D", "", explanation)
    return False


def _description_excerpt(description):
    candidates = [part.strip(" \n\t.!?") for part in re.split(r"[.!?]+", description)]
    candidates = [part for part in candidates if len(part.split()) >= 4]
    for candidate in candidates:
        intro = candidate.casefold()
        if not intro.startswith(("я ", "я,", "меня зовут", "привет")) and " — " not in candidate[:60]:
            return " ".join(candidate.split()[:24])
    if candidates:
        return " ".join(candidates[0].split()[:24])
    return "Описание подтверждает заявленный формат работы"


def _sentence_count(text):
    return len([part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part])


def _extract_output_text(api_response):
    for output in api_response.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"]
    raise ValueError("OpenAI response did not contain output text")
