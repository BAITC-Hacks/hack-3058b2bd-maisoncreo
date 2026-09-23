import os
import unittest
from unittest.mock import patch

from explanation_layer import (
    ExplanationValidationError,
    enrich_recommendations,
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
        "explanation": (
            f"Исполнитель {contractor_id}: проводит корпоративы на языке «русский». "
            "В описании указано: «Работает с деловой аудиторией»."
        ),
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

    def test_generic_or_unqualified_price_claim_uses_fallback(self):
        output = valid_model_output()
        output["explanation"] = (
            "Исполнитель c1 — отличный выбор за 200000 KZT. "
            "В описании указано: «Работает с деловой аудиторией»."
        )
        output["structured_evidence"] = ["price_from_kzt"]

        enriched = enrich_recommendations(
            result("c1"), [contractor()], QUERY, provider=lambda *_: output
        )

        self.assertEqual(enriched["recommendations"][0]["explanation_source"], "fallback")
        self.assertIn("стартовая цена — от", enriched["recommendations"][0]["explanation"])

    def test_provider_failure_uses_fallback(self):
        def failing_provider(*_):
            raise TimeoutError("API unavailable")

        enriched = enrich_recommendations(
            result("c1"), [contractor()], QUERY, provider=failing_provider
        )

        self.assertEqual(enriched["recommendations"][0]["explanation_source"], "fallback")
        evidence = enriched["recommendations"][0]["explanation_evidence"]["description_excerpt"]
        self.assertIn(evidence, enriched["recommendations"][0]["explanation"])

    def test_no_api_key_uses_fallback_without_api_call(self):
        with patch.dict(os.environ, {}, clear=True):
            enriched = enrich_recommendations(result("c1"), [contractor()], QUERY)

        self.assertEqual(enriched["recommendations"][0]["explanation_source"], "fallback")

    def test_enrichment_never_changes_selection_or_order(self):
        records = [contractor("c1"), contractor("c2"), contractor("c3")]

        enriched = enrich_recommendations(result("c3", "c1", "c2"), records, QUERY)

        self.assertEqual(
            [card["id"] for card in enriched["recommendations"]],
            ["c3", "c1", "c2"],
        )

    def test_validator_rejects_availability_claim(self):
        output = valid_model_output()
        output["explanation"] = (
            "Исполнитель c1 доступен на выбранную дату и проводит корпоративы. "
            "В описании указано: «Работает с деловой аудиторией»."
        )
        output["structured_evidence"] = ["event_format"]

        with self.assertRaises(ValueError):
            validate_explanation(output, contractor(), QUERY)

    def test_description_name_cannot_replace_anonymous_name(self):
        output = valid_model_output()
        output["explanation"] = (
            "Другое имя проводит корпоративы на языке «русский». "
            "В описании указано: «Работает с деловой аудиторией»."
        )

        enriched = enrich_recommendations(
            result("c1"), [contractor()], QUERY, provider=lambda *_: output
        )

        card = enriched["recommendations"][0]
        self.assertEqual(card["explanation_source"], "fallback")
        self.assertTrue(card["explanation"].startswith("Исполнитель c1"))


if __name__ == "__main__":
    unittest.main()
