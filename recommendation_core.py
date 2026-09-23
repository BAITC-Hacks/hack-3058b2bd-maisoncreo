"""Deterministic contractor eligibility and recommendation logic."""

from datetime import date


MATCHED = "MATCHED"
CATEGORY_NOT_FOUND = "CATEGORY_NOT_FOUND"
NO_ELIGIBLE_CANDIDATES = "NO_ELIGIBLE_CANDIDATES"
UNKNOWN_AVAILABILITY = "UNKNOWN_AVAILABILITY"
INVALID_REQUEST = "INVALID_REQUEST"

NON_TIME_CATEGORIES = frozenset(
    {"Флорист", "Декоратор", "Подарки и сувениры"}
)
REQUIRED_QUERY_FIELDS = (
    "city",
    "event_date",
    "event_format",
    "category",
    "budget_kzt",
)


def recommend(records, query, availability_horizon):
    """Return up to three deterministic recommendations from CSV-shaped records."""
    records = list(records)
    validation_error = _validate_request(query, availability_horizon)
    if validation_error:
        return _response(INVALID_REQUEST, [], validation_error)

    event_date = date.fromisoformat(query["event_date"])
    horizon_start = date.fromisoformat(availability_horizon["start"])
    horizon_end = date.fromisoformat(availability_horizon["end"])
    if not horizon_start <= event_date <= horizon_end:
        return _response(
            UNKNOWN_AVAILABILITY,
            [],
            "The requested date is outside the known availability horizon.",
        )

    requested_category = query["category"]
    known_categories = {
        category
        for record in records
        for category in _tokens(record.get("categories"))
    }
    if requested_category not in known_categories:
        return _response(
            CATEGORY_NOT_FOUND,
            [],
            f"Category {requested_category!r} is not present in the contractor data.",
        )

    eligible = [record for record in records if _is_eligible(record, query)]
    eligible.sort(key=_ranking_key)
    selected = eligible[:3]
    recommendations = [_card(record, query) for record in selected]

    if not recommendations:
        return _response(
            NO_ELIGIBLE_CANDIDATES,
            [],
            "No contractors met all hard constraints, including availability.",
        )

    if len(eligible) < 3:
        message = (
            f"Found {len(eligible)} eligible contractor(s); returning all because "
            f"only {len(eligible)} met every hard constraint, including availability."
        )
    else:
        message = f"Returning {len(recommendations)} of {len(eligible)} eligible contractors."

    return _response(MATCHED, recommendations, message)


def _validate_request(query, availability_horizon):
    if not isinstance(query, dict):
        return "Query must be a mapping."

    missing = [field for field in REQUIRED_QUERY_FIELDS if query.get(field) in (None, "")]
    if missing:
        return f"Missing required field(s): {', '.join(missing)}."

    try:
        requested_date = date.fromisoformat(query["event_date"])
        horizon_start = date.fromisoformat(availability_horizon["start"])
        horizon_end = date.fromisoformat(availability_horizon["end"])
    except (KeyError, TypeError, ValueError):
        return "Dates and availability horizon must use valid ISO dates."

    if horizon_start > horizon_end:
        return "Availability horizon start must not be after its end."

    if not isinstance(requested_date, date):
        return "Event date is invalid."

    try:
        budget = int(query["budget_kzt"])
    except (TypeError, ValueError):
        return "Budget must be an integer amount in KZT."
    if budget < 0:
        return "Budget must not be negative."

    duration = query.get("duration_hours")
    if duration is not None:
        try:
            if float(duration) <= 0:
                return "Duration must be greater than zero."
        except (TypeError, ValueError):
            return "Duration must be numeric."

    return None


def _is_eligible(record, query):
    requested_category = query["category"]
    if requested_category not in _tokens(record.get("categories")):
        return False
    if record.get("city") != query["city"]:
        return False
    if query["event_format"] not in _tokens(record.get("event_formats")):
        return False

    try:
        if int(record["price_from_kzt"]) > int(query["budget_kzt"]):
            return False
    except (KeyError, TypeError, ValueError):
        return False

    requested_language = query.get("language")
    if requested_language and requested_language not in _tokens(record.get("languages")):
        return False

    requested_duration = query.get("duration_hours")
    if requested_duration is not None and requested_category not in NON_TIME_CATEGORIES:
        max_hours = record.get("max_hours")
        if max_hours in (None, ""):
            return False
        try:
            if float(max_hours) < float(requested_duration):
                return False
        except (TypeError, ValueError):
            return False

    if query["event_date"] in _tokens(record.get("busy_dates")):
        return False

    return True


def _ranking_key(record):
    """Prefer lower starting price, then use stable ID as the final tie-breaker."""
    return int(record["price_from_kzt"]), str(record["id"])


def _card(record, query):
    price = int(record["price_from_kzt"])
    name = record.get("anon_name", "Unnamed contractor")
    first_sentence = (
        f"{name}: формат «{query['event_format']}» в городе {query['city']}; "
        f"стартовая цена {price:,} KZT укладывается в бюджет "
        f"{int(query['budget_kzt']):,} KZT."
    )

    fit_details = [f"доступен на дату {query['event_date']}"]
    if query.get("language"):
        fit_details.append(f"работает на языке «{query['language']}»")
    if (
        query.get("duration_hours") is not None
        and query["category"] not in NON_TIME_CATEGORIES
    ):
        fit_details.append(f"максимальная длительность — {record['max_hours']} ч")
    elif query.get("duration_hours") is not None:
        fit_details.append("длительность к этой категории не применяется")

    return {
        "id": record["id"],
        "name": name,
        "category": record.get("categories", ""),
        "city": record.get("city", ""),
        "price_from_kzt": price,
        "explanation": first_sentence + " Подрядчик " + ", ".join(fit_details) + ".",
    }


def _tokens(value):
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return tuple(token.strip() for token in value.split("|") if token.strip())
    return tuple(value)


def _response(status, recommendations, message):
    return {
        "status": status,
        "recommendations": recommendations,
        "message": message,
    }
