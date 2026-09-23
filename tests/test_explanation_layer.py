import os
import unittest
from unittest.mock import patch

from explanation_layer import (
    ExplanationValidationError,
    enrich_recommendations,
    fallback_explanation,
    validate_explanation,
)


QUERY = {
    "city": "Алматы",
    "event_date": "2030-01-15",
    "event_format": "корпоратив",
    "category": "Ведущий",
    "budget_kzt": 300_000,
    "language": "русский",
    "duration_hours": 4,
}


def contractor(contractor_id="c1"):
    return {
        "id": contractor_id,
        "anon_name": f"Исполнитель {contractor_id}",
        "categories": "Ведущий",
        "city": "Алматы",
        "price_from_kzt": 200_000,
        "event_formats": "корпоратив",
        "languages": "русский",
        "max_hours": 6,
        "description": "Создаёт авторские сценарии. Работает с деловой аудиторией.",
    }


def result(*ids):
    return {
        "status": "MATCHED",
        "recommendations": [
            {"id": contractor_id, "explanation": "core fallback"}
            for contractor_id in ids
        ],
        "message": "matched",
    }


def valid_model_output(contractor_id="c1"):
    return {
        "description_evidence": "Работает с деловой аудиторией",
        "structured_evidence": ["event_format", "language"],
    }


class ExplanationLayerTests(unittest.TestCase):
    def test_valid_model_output_is_used_and_evidence_is_preserved(self):
        enriched = enrich_recommendations(
            result("c1"), [contractor()], QUERY, provider=lambda *_: valid_model_output()
        )

        card = enriched["recommendations"][0]
        self.assertEqual(card["explanation_source"], "openai")
        self.assertEqual(
            card["explanation_evidence"]["description_excerpt"],
            "Работает с деловой аудиторией",
        )
        self.assertIn("формата «корпоратив»", card["explanation"])
        self.assertIn("язык «русский»", card["explanation"])

    def test_unsupported_description_evidence_uses_fallback(self):
        output = valid_model_output()
        output["description_evidence"] = "Проводит концерты на стадионах"

        enriched = enrich_recommendations(
            result("c1"), [contractor()], QUERY, provider=lambda *_: output
        )

        self.assertEqual(enriched["recommendations"][0]["explanation_source"], "fallback")

    def test_validator_rejects_decorated_description_evidence(self):
        output = valid_model_output()
        output["description_evidence"] = "«Работает с деловой аудиторией»"

        with self.assertRaisesRegex(
            ExplanationValidationError,
            "Description evidence is not an exact source excerpt",
        ):
            validate_explanation(output, contractor(), QUERY)

    def test_model_authored_fabricated_claim_never_reaches_explanation(self):
        output = valid_model_output()
        output["explanation"] = "Исполнитель c1 имеет рейтинг 5.0."

        enriched = enrich_recommendations(
            result("c1"), [contractor()], QUERY, provider=lambda *_: output
        )

        self.assertEqual(enriched["recommendations"][0]["explanation_source"], "fallback")
        self.assertNotIn("рейтинг 5.0", enriched["recommendations"][0]["explanation"])

    def test_provider_failure_uses_fallback(self):
        def failing_provider(*_):
            raise TimeoutError("API unavailable")

        enriched = enrich_recommendations(
            result("c1"), [contractor()], QUERY, provider=failing_provider
        )

        self.assertEqual(enriched["recommendations"][0]["explanation_source"], "fallback")
        evidence = enriched["recommendations"][0]["explanation_evidence"]["description_excerpt"]
        self.assertIn(evidence, enriched["recommendations"][0]["explanation"])

    def test_provider_failure_prevents_subsequent_provider_calls(self):
        calls = []

        def failing_provider(contractor_record, _):
            calls.append(contractor_record["id"])
            raise TimeoutError("API unavailable")

        records = [contractor("c1"), contractor("c2"), contractor("c3")]
        enriched = enrich_recommendations(
            result("c1", "c2", "c3"), records, QUERY, provider=failing_provider
        )

        self.assertEqual(calls, ["c1"])
        self.assertEqual(
            [card["explanation_source"] for card in enriched["recommendations"]],
            ["fallback", "fallback", "fallback"],
        )

    def test_no_api_key_uses_fallback_without_api_call(self):
        with patch.dict(os.environ, {}, clear=True):
            enriched = enrich_recommendations(result("c1"), [contractor()], QUERY)

        self.assertEqual(enriched["recommendations"][0]["explanation_source"], "fallback")

    def test_enrichment_never_changes_selection_or_order(self):
        records = [contractor("c1"), contractor("c2"), contractor("c3")]

        enriched = enrich_recommendations(
            result("c3", "c1", "c2"),
            records,
            QUERY,
            provider=lambda contractor_record, _: valid_model_output(contractor_record["id"]),
        )

        self.assertEqual(
            [card["id"] for card in enriched["recommendations"]],
            ["c3", "c1", "c2"],
        )

    def test_validator_rejects_unsupported_structured_evidence(self):
        output = valid_model_output()
        output["structured_evidence"] = ["max_hours"]
        query = dict(QUERY, duration_hours=7)

        with self.assertRaisesRegex(ExplanationValidationError, "is not grounded"):
            validate_explanation(output, contractor(), query)

    def test_empty_description_fallback_is_structured_only(self):
        record = contractor()
        record["description"] = ""

        generated = fallback_explanation(record, QUERY)

        self.assertEqual(generated["description_evidence"], "")
        self.assertNotIn("В описании", generated["explanation"])
        self.assertIn("формата «корпоратив»", generated["explanation"])

    def test_fallback_description_evidence_is_an_exact_source_substring(self):
        record = contractor()
        record["description"] = "  Коротко.  Точный   текст с исходными пробелами!  "

        generated = fallback_explanation(record, QUERY)

        self.assertTrue(generated["description_evidence"])
        self.assertIn(generated["description_evidence"], record["description"])


if __name__ == "__main__":
    unittest.main()
