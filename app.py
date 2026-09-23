"""Streamlit demo for deterministic contractor recommendations."""

import csv
from datetime import date
from pathlib import Path

import streamlit as st

from explanation_layer import enrich_recommendations
from recommendation_core import (
    CATEGORY_NOT_FOUND,
    INVALID_REQUEST,
    MATCHED,
    NO_ELIGIBLE_CANDIDATES,
    UNKNOWN_AVAILABILITY,
    recommend,
)


DATA_PATH = Path(__file__).parent / "data" / "contractors.csv"
NO_LANGUAGE = "Не важно"

PRESETS = {
    "DENSE": {
        "city": "Алматы",
        "event_date": date(2026, 9, 30),
        "event_format": "корпоратив",
        "category": "Ведущий",
        "budget_kzt": 1_000_000,
        "duration_hours": 6,
        "language": "русский",
    },
    "RARE": {
        "city": "Алматы",
        "event_date": date(2026, 9, 24),
        "event_format": "свадьба",
        "category": "Ведущий церемонии",
        "budget_kzt": 300_000,
        "duration_hours": 3,
        "language": "русский",
    },
    "NO RESULT": {
        "city": "Астана",
        "event_date": date(2026, 9, 28),
        "event_format": "свадьба",
        "category": "Фотограф",
        "budget_kzt": 1_000_000,
        "duration_hours": 8,
        "language": "русский",
    },
}
PRESET_LABELS = {
    "DENSE": "Много вариантов",
    "RARE": "Мало вариантов",
    "NO RESULT": "Нет подходящих",
}


@st.cache_data
def load_records():
    with DATA_PATH.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def taxonomy(records, field):
    return sorted(
        {
            token.strip()
            for record in records
            for token in record[field].split("|")
            if token.strip()
        }
    )


def availability_horizon(records):
    dates = [
        value
        for record in records
        for value in record["busy_dates"].split("|")
        if value
    ]
    return {"start": min(dates), "end": max(dates)}


def apply_preset(preset_name):
    st.session_state.update(PRESETS[preset_name])
    st.session_state["run_preset"] = True


def as_bool(value):
    return value is True or str(value).casefold() == "true"


def format_kzt(value):
    return f"{int(value):,}".replace(",", " ") + " ₸"


def build_query():
    query = {
        "city": st.session_state["city"],
        "event_date": st.session_state["event_date"].isoformat(),
        "event_format": st.session_state["event_format"],
        "category": st.session_state["category"],
        "budget_kzt": int(st.session_state["budget_kzt"]),
    }
    if st.session_state["duration_hours"] > 0:
        query["duration_hours"] = int(st.session_state["duration_hours"])
    if st.session_state["language"] != NO_LANGUAGE:
        query["language"] = st.session_state["language"]
    return query


def render_card(card, record, query):
    with st.container(border=True):
        st.subheader(card["name"])
        st.metric("Стартовая стоимость", f"от {format_kzt(card['price_from_kzt'])}")

        facts = [
            f"**Категория:** {query['category']}",
            f"**Город:** {card['city']}",
            f"**Формат:** {query['event_format']}",
            f"**Дата:** {query['event_date']}",
        ]
        if query.get("language"):
            facts.append(f"**Язык:** {query['language']}")
        if query.get("duration_hours") is not None and record.get("max_hours"):
            facts.append(f"**До:** {record['max_hours']} ч")
        st.markdown("  \n".join(facts))

        st.write(card["explanation"])

        indicators = []
        if as_bool(record.get("synthetic")):
            indicators.append("синтетическая запись")
        if as_bool(record.get("city_imputed")):
            indicators.append("город восстановлен")
        if as_bool(record.get("price_imputed")):
            indicators.append("цена восстановлена")
        if indicators:
            st.caption("Данные: " + " · ".join(indicators))

        source_label = (
            "Объяснение с помощью OpenAI"
            if card.get("explanation_source") == "openai"
            else "Проверенное резервное объяснение"
        )
        st.caption(f"Источник объяснения: {source_label}")
        evidence = card.get("explanation_evidence", {})
        if evidence:
            with st.expander("Показать основание"):
                st.write(evidence.get("description_excerpt", ""))
                fields = ", ".join(evidence.get("structured_fields", []))
                st.caption(f"Проверенные поля: {fields}")


def render_result(result, records, query):
    status = result["status"]
    if status == MATCHED:
        enriched = enrich_recommendations(result, records, query)
        recommendation_count = len(enriched["recommendations"])
        if recommendation_count < 3:
            if recommendation_count == 1:
                shortage_message = "Найден 1 подходящий подрядчик."
            else:
                shortage_message = "Найдено 2 подходящих подрядчика."
            st.info(
                shortage_message
                + " Показываем всех, кто соответствует условиям и доступен "
                "на выбранную дату."
            )
        records_by_id = {record["id"]: record for record in records}
        st.subheader("Рекомендации")
        for card in enriched["recommendations"]:
            render_card(card, records_by_id[card["id"]], query)
        st.caption("Рекомендация не является бронированием или подтверждением заказа.")
    elif status == CATEGORY_NOT_FOUND:
        st.warning("Такой категории нет в каталоге подрядчиков. Выберите другую категорию.")
    elif status == NO_ELIGIBLE_CANDIDATES:
        st.warning(
            "Подрядчики этой категории есть, но на выбранную дату никто не "
            "соответствует всем заданным условиям."
        )
    elif status == UNKNOWN_AVAILABILITY:
        st.error("Выбранная дата находится вне известного периода доступности.")
    elif status == INVALID_REQUEST:
        st.error("Проверьте обязательные поля и корректность введённых значений.")
    else:
        st.error("Не удалось обработать запрос.")


def main():
    st.set_page_config(page_title="Smart Contractor Selection", page_icon="✨")
    records = load_records()
    horizon = availability_horizon(records)
    horizon_start = date.fromisoformat(horizon["start"])
    horizon_end = date.fromisoformat(horizon["end"])

    st.title("Умный подбор подрядчиков")
    st.write("До трёх проверенных рекомендаций по вашим условиям — без бронирования.")

    st.caption("Демо-запросы")
    preset_columns = st.columns(3)
    for column, preset_name in zip(preset_columns, PRESETS):
        column.button(
            PRESET_LABELS[preset_name],
            use_container_width=True,
            on_click=apply_preset,
            args=(preset_name,),
        )

    defaults = PRESETS["DENSE"]
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    cities = taxonomy(records, "city")
    formats = taxonomy(records, "event_formats")
    categories = taxonomy(records, "categories")
    languages = [NO_LANGUAGE] + taxonomy(records, "languages")

    with st.form("contractor_request"):
        left, right = st.columns(2)
        with left:
            st.selectbox("Город", cities, key="city")
            st.date_input(
                "Дата мероприятия",
                min_value=horizon_start,
                max_value=horizon_end,
                key="event_date",
                help=f"Доступность известна с {horizon['start']} по {horizon['end']}.",
            )
            st.selectbox("Формат мероприятия", formats, key="event_format")
            st.selectbox("Категория подрядчика", categories, key="category")
        with right:
            st.number_input(
                "Бюджет, KZT",
                min_value=0,
                step=50_000,
                key="budget_kzt",
                help="Сравнивается со стартовой ценой подрядчика.",
            )
            st.number_input(
                "Длительность, часов (0 — не указана)",
                min_value=0,
                max_value=24,
                step=1,
                key="duration_hours",
            )
            st.selectbox("Язык", languages, key="language")

        submitted = st.form_submit_button(
            "Подобрать подрядчиков", type="primary", use_container_width=True
        )

    if submitted or st.session_state.pop("run_preset", False):
        query = build_query()
        result = recommend(records, query, availability_horizon=horizon)
        render_result(result, records, query)


if __name__ == "__main__":
    main()
