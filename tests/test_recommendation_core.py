import unittest


HORIZON = {"start": "2030-01-01", "end": "2030-01-31"}


def contractor(contractor_id, **overrides):
    record = {
        "id": contractor_id,
        "anon_name": f"Contractor {contractor_id}",
        "categories": "Ведущий",
        "city": "Алматы",
        "city_imputed": False,
        "synthetic": False,
        "price_from_kzt": 100_000,
        "price_imputed": False,
        "event_formats": "свадьба|корпоратив",
        "languages": "русский",
        "max_hours": 6,
        "busy_dates": "",
        "description": "Проводит свадьбы и корпоративные мероприятия.",
    }
    record.update(overrides)
    return record


def query(**overrides):
    request = {
        "city": "Алматы",
        "event_date": "2030-01-15",
        "event_format": "свадьба",
        "category": "Ведущий",
        "budget_kzt": 100_000,
    }
    request.update(overrides)
    return request


def recommend(records, request):
    from recommendation_core import recommend as recommend_contractors

    return recommend_contractors(records, request, availability_horizon=HORIZON)


def recommendation_ids(result):
    return [item["id"] for item in result["recommendations"]]


class RecommendationCoreAcceptanceTests(unittest.TestCase):
    def test_busy_contractor_is_never_returned(self):
        result = recommend(
            [
                contractor("busy", busy_dates="2030-01-15"),
                contractor("free"),
            ],
            query(),
        )

        self.assertEqual(recommendation_ids(result), ["free"])

    def test_wrong_city_is_never_returned(self):
        result = recommend(
            [contractor("wrong", city="Астана"), contractor("right")], query()
        )

        self.assertEqual(recommendation_ids(result), ["right"])

    def test_wrong_category_is_never_returned(self):
        result = recommend(
            [contractor("wrong", categories="Фотограф"), contractor("right")],
            query(),
        )

        self.assertEqual(recommendation_ids(result), ["right"])

    def test_unsupported_event_format_is_never_returned(self):
        result = recommend(
            [
                contractor("wrong", event_formats="конференция"),
                contractor("right"),
            ],
            query(),
        )

        self.assertEqual(recommendation_ids(result), ["right"])

    def test_starting_price_above_budget_is_rejected(self):
        result = recommend(
            [contractor("expensive", price_from_kzt=100_001)], query()
        )

        self.assertEqual(result["status"], "NO_ELIGIBLE_CANDIDATES")
        self.assertEqual(result["recommendations"], [])

    def test_starting_price_equal_to_budget_qualifies(self):
        result = recommend(
            [contractor("exact", price_from_kzt=100_000)], query()
        )

        self.assertEqual(recommendation_ids(result), ["exact"])

    def test_requested_language_is_enforced(self):
        result = recommend(
            [
                contractor("wrong", languages="казахский"),
                contractor("right", languages="казахский|русский"),
            ],
            query(language="русский"),
        )

        self.assertEqual(recommendation_ids(result), ["right"])

    def test_language_is_not_a_filter_when_omitted(self):
        result = recommend([contractor("kazakh", languages="казахский")], query())

        self.assertEqual(recommendation_ids(result), ["kazakh"])

    def test_insufficient_max_hours_is_rejected_for_time_based_category(self):
        result = recommend(
            [contractor("short", max_hours=5)], query(duration_hours=6)
        )

        self.assertEqual(result["status"], "NO_ELIGIBLE_CANDIDATES")
        self.assertEqual(result["recommendations"], [])

    def test_max_hours_equal_to_requested_duration_qualifies(self):
        result = recommend(
            [contractor("exact", max_hours=6)], query(duration_hours=6)
        )

        self.assertEqual(recommendation_ids(result), ["exact"])

    def test_missing_max_hours_is_not_applicable_for_non_time_categories(self):
        for category in ("Флорист", "Декоратор", "Подарки и сувениры"):
            with self.subTest(category=category):
                result = recommend(
                    [contractor("eligible", categories=category, max_hours=None)],
                    query(category=category, duration_hours=8),
                )

                self.assertEqual(recommendation_ids(result), ["eligible"])

    def test_at_most_three_recommendations_are_returned(self):
        records = [contractor(f"c{number}") for number in range(5)]

        result = recommend(records, query())

        self.assertEqual(len(result["recommendations"]), 3)

    def test_same_query_produces_exactly_the_same_ordering(self):
        records = [contractor(f"c{number}") for number in range(5)]

        first = recommend(records, query())
        second = recommend(records, query())

        self.assertEqual(recommendation_ids(first), recommendation_ids(second))

    def test_category_not_found_is_distinct_from_no_eligible_candidates(self):
        records = [contractor("host", city="Астана")]

        missing_category = recommend(records, query(category="Фотограф"))
        no_eligible = recommend(records, query(category="Ведущий"))

        self.assertEqual(missing_category["status"], "CATEGORY_NOT_FOUND")
        self.assertEqual(no_eligible["status"], "NO_ELIGIBLE_CANDIDATES")

    def test_fewer_than_three_eligible_contractors_returns_all_of_them(self):
        result = recommend([contractor("one"), contractor("two")], query())

        self.assertEqual(set(recommendation_ids(result)), {"one", "two"})
        self.assertEqual(len(result["recommendations"]), 2)

    def test_date_outside_horizon_is_rejected_as_unknown_availability(self):
        result = recommend(
            [contractor("apparently-free")], query(event_date="2030-02-01")
        )

        self.assertEqual(result["status"], "UNKNOWN_AVAILABILITY")
        self.assertEqual(result["recommendations"], [])

    def test_multi_category_contractor_matches_any_contained_category(self):
        result = recommend(
            [contractor("multi", categories="Фотограф|Видеограф")],
            query(category="Видеограф"),
        )

        self.assertEqual(recommendation_ids(result), ["multi"])


if __name__ == "__main__":
    unittest.main()
