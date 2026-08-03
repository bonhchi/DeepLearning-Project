"""Deterministic intent-aware query rewriting for the NLP search MVP."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final, TypedDict

from src.nlp.entity_extractor import (
    CATEGORY_ALIASES,
    COLOR_ALIASES,
    FEATURE_ALIASES,
    MATERIAL_ALIASES,
    PURPOSE_ALIASES,
    EntityExtractor,
    ExtractedEntities,
    normalize_text,
)


LOGGER = logging.getLogger(__name__)


class PreferenceDecision(TypedDict):
    """Explain why one profile preference was added or ignored."""

    field: str
    value: object
    reason: str


class QueryRewriteResult(TypedDict):
    """Stable output contract for one rewritten query."""

    original_query: str
    rewritten_query: str
    added_preferences: list[PreferenceDecision]
    ignored_preferences: list[PreferenceDecision]


SUPPORTED_INTENTS: Final[set[str]] = {
    "product_search",
    "need_based_search",
    "similar_product_search",
    "personalized_recommendation",
    "availability_check",
    "comparison",
}
PERSONALIZABLE_INTENTS: Final[set[str]] = {
    "product_search",
    "need_based_search",
    "personalized_recommendation",
}

_INTENT_ALIASES: Final[dict[str, str]] = {
    "availability": "availability_check",
    "compare": "comparison",
    "comparison_search": "comparison",
    "need_search": "need_based_search",
    "personalized": "personalized_recommendation",
    "recommendation": "personalized_recommendation",
    "search": "product_search",
    "similar": "similar_product_search",
}

_PROFILE_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "category": ("preferred_categories", "categories"),
    "brand": ("preferred_brands", "brands"),
    "color": ("preferred_colors", "colors"),
    "size": ("preferred_sizes", "sizes"),
    "material": ("preferred_materials", "materials"),
    "feature": ("preferred_features", "features", "attributes"),
    "purpose": ("preferred_purposes", "purposes"),
    "preference": ("positive_preferences",),
}

_SCALAR_CONSTRAINT_FIELDS: Final[set[str]] = {
    "category",
    "brand",
    "color",
    "size",
    "material",
}

_OPPOSITE_VALUES: Final[dict[str, set[str]]] = {
    "wired": {"wireless"},
    "wireless": {"wired"},
    "waterproof": {"not waterproof", "non waterproof"},
    "not waterproof": {"waterproof"},
    "new": {"used", "refurbished"},
    "used": {"new"},
    "refurbished": {"new"},
}

_VI_FEATURE_LABELS: Final[dict[str, str]] = {
    "anti_slip": "chống trượt",
    "bluetooth": "Bluetooth",
    "breathable": "thoáng khí",
    "durable": "bền",
    "fast_charging": "sạc nhanh",
    "foldable": "gấp gọn",
    "lightweight": "nhẹ",
    "long_battery_life": "pin lâu",
    "noise_cancelling": "chống ồn",
    "organic": "hữu cơ",
    "touchscreen": "cảm ứng",
    "waterproof": "chống nước",
    "wired": "có dây",
    "wireless": "không dây",
}
_VI_PURPOSE_LABELS: Final[dict[str, str]] = {
    "commuting": "đi lại hằng ngày",
    "gaming": "chơi game",
    "gift": "làm quà tặng",
    "office": "văn phòng",
    "outdoor": "ngoài trời",
    "running": "chạy bộ",
    "skincare": "chăm sóc da",
    "sports": "thể thao",
    "study": "học tập",
    "travel": "du lịch",
    "work": "làm việc",
    "workout": "tập luyện",
}


def _normalize_intent(intent: object) -> str:
    if isinstance(intent, Mapping):
        intent = intent.get("intent", intent.get("label", ""))
    normalized = normalize_text(str(intent or "product_search")).strip()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return _INTENT_ALIASES.get(normalized, normalized)


def _normalized_value(value: object) -> str:
    return " ".join(normalize_text(str(value)).replace("_", " ").split())


def _iter_profile_values(raw_value: object) -> list[object]:
    if raw_value is None:
        return []
    if isinstance(raw_value, Mapping):
        if "value" in raw_value:
            return [raw_value["value"]]
        # Profiles often store preference -> interaction weight mappings.
        sortable: list[tuple[float, object]] = []
        for key, score in raw_value.items():
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                numeric_score = 0.0
            sortable.append((numeric_score, key))
        return [key for _, key in sorted(sortable, key=lambda item: -item[0])]
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        return list(raw_value)
    return [raw_value]


def _canonicalize(
    field: str,
    value: object,
) -> object:
    if not isinstance(value, str):
        return value
    dictionaries = {
        "category": CATEGORY_ALIASES,
        "color": COLOR_ALIASES,
        "material": MATERIAL_ALIASES,
        "feature": FEATURE_ALIASES,
        "purpose": PURPOSE_ALIASES,
    }
    dictionary = dictionaries.get(field)
    if dictionary is None:
        return value.strip()
    normalized = _normalized_value(value)
    for canonical, aliases in dictionary.items():
        candidates = {
            _normalized_value(canonical),
            *(_normalized_value(alias) for alias in aliases),
        }
        if normalized in candidates:
            return canonical
    return value.strip()


def _entity_payload(entities: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if entities is None:
        return {}
    nested = entities.get("entities")
    return nested if isinstance(nested, Mapping) else entities


def _entity_entries(entities: Mapping[str, Any], field: str) -> list[Mapping[str, Any]]:
    raw = entities.get(field)
    if raw is None:
        return []
    values: Iterable[object]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = raw
    else:
        values = [raw]
    output: list[Mapping[str, Any]] = []
    for value in values:
        if isinstance(value, Mapping):
            output.append(value)
        else:
            output.append({"value": value})
    return output


def _entity_values(entities: Mapping[str, Any], field: str) -> list[tuple[object, bool]]:
    return [
        (entry.get("value"), bool(entry.get("negated", False)))
        for entry in _entity_entries(entities, field)
        if entry.get("value") is not None
    ]


def _collect_profile_preferences(profile: Mapping[str, Any]) -> list[tuple[str, object]]:
    output: list[tuple[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for field, keys in _PROFILE_KEYS.items():
        for key in keys:
            if key not in profile:
                continue
            for raw_value in _iter_profile_values(profile.get(key)):
                if raw_value is None or str(raw_value).strip() == "":
                    continue
                resolved_field = field
                resolved_value = raw_value
                if field == "preference" and isinstance(raw_value, str) and ":" in raw_value:
                    prefix, payload = raw_value.split(":", 1)
                    resolved_field = {
                        "attribute": "feature",
                        "brand": "brand",
                        "category": "category",
                        "color": "color",
                        "feature": "feature",
                        "material": "material",
                        "purpose": "purpose",
                        "size": "size",
                    }.get(_normalized_value(prefix), "preference")
                    resolved_value = (
                        payload.strip() if resolved_field != "preference" else raw_value
                    )
                value = _canonicalize(resolved_field, resolved_value)
                marker = (resolved_field, _normalized_value(value))
                if marker not in seen:
                    seen.add(marker)
                    output.append((resolved_field, value))
    return output


def _collect_negative_profile_preferences(
    profile: Mapping[str, Any],
) -> list[tuple[str, object]]:
    """Return structured exclusions without treating them as positive terms."""

    output: list[tuple[str, object]] = []
    seen: set[tuple[str, str]] = set()
    field_aliases = {
        "attribute": "feature",
        "brand": "brand",
        "category": "category",
        "color": "color",
        "feature": "feature",
        "material": "material",
        "purpose": "purpose",
        "review term": "review_term",
        "review_term": "review_term",
        "size": "size",
    }
    for raw_value in _iter_profile_values(profile.get("negative_preferences")):
        if raw_value is None or not str(raw_value).strip():
            continue
        field = "preference"
        value: object = raw_value
        if isinstance(raw_value, str) and ":" in raw_value:
            prefix, payload = raw_value.split(":", 1)
            field = field_aliases.get(_normalized_value(prefix), "preference")
            value = payload.strip() if field != "preference" else raw_value.strip()
        value = _canonicalize(field, value)
        marker = (field, _normalized_value(value))
        if marker not in seen:
            seen.add(marker)
            output.append((field, value))
    return output


def _profile_price_range(profile: Mapping[str, Any]) -> dict[str, object] | None:
    raw = profile.get("preferred_price_range", profile.get("price_range"))
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        minimum = raw.get("min_price", raw.get("min"))
        maximum = raw.get("max_price", raw.get("max"))
        if minimum is None and maximum is None and "value" in raw:
            return {"label": raw["value"]}
        return {
            "min_price": minimum,
            "max_price": maximum,
            "currency": str(raw.get("currency", "UNKNOWN")).upper(),
        }
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
        if len(values) >= 2:
            return {
                "min_price": values[0],
                "max_price": values[1],
                "currency": "UNKNOWN",
            }
    return {"label": raw}


def _conflict_reason(
    field: str,
    value: object,
    entities: Mapping[str, Any],
    normalized_query: str,
) -> str | None:
    candidate = _normalized_value(value)
    explicit = _entity_values(entities, field)
    for explicit_value, negated in explicit:
        normalized_explicit = _normalized_value(explicit_value)
        if candidate == normalized_explicit:
            return "excluded_by_query" if negated else "already_explicit_in_query"
        if field in _SCALAR_CONSTRAINT_FIELDS and not negated:
            return "conflicts_with_explicit_query_constraint"
        if candidate in _OPPOSITE_VALUES.get(normalized_explicit, set()):
            return "conflicts_with_explicit_query_constraint"
        if normalized_explicit in _OPPOSITE_VALUES.get(candidate, set()):
            return "conflicts_with_explicit_query_constraint"

    # Free-form positive preferences still respect direct exclusions.
    if f"not {candidate}" in normalized_query or f"without {candidate}" in normalized_query:
        return "excluded_by_query"
    if f"khong {candidate}" in normalized_query and candidate != "day":
        return "excluded_by_query"

    if field == "preference":
        all_explicit = [
            (_normalized_value(explicit_value), negated)
            for entity_field in (
                "category",
                "brand",
                "color",
                "size",
                "material",
                "feature",
                "purpose",
            )
            for explicit_value, negated in _entity_values(entities, entity_field)
        ]
        for explicit_value, negated in all_explicit:
            if candidate == explicit_value:
                return "excluded_by_query" if negated else "already_explicit_in_query"
            if candidate in _OPPOSITE_VALUES.get(explicit_value, set()):
                return "conflicts_with_explicit_query_constraint"
    return None


def _negative_conflict_reason(
    field: str,
    value: object,
    entities: Mapping[str, Any],
) -> str | None:
    """Give explicit query constraints precedence over historical dislikes."""

    if field in {"preference", "review_term"}:
        # Free-form dislikes can be sentiment or review prose.  Turning them
        # into Boolean query constraints is too error-prone.
        return "unsafe_unstructured_negative_preference"
    candidate = _normalized_value(value)
    for explicit_value, negated in _entity_values(entities, field):
        normalized_explicit = _normalized_value(explicit_value)
        if candidate == normalized_explicit:
            return (
                "already_explicit_in_query"
                if negated
                else "explicit_query_overrides_negative_profile"
            )
        opposites = _OPPOSITE_VALUES.get(candidate, set())
        reverse_opposites = _OPPOSITE_VALUES.get(normalized_explicit, set())
        if (
            normalized_explicit in opposites
            or candidate in reverse_opposites
        ):
            return (
                "explicit_query_overrides_negative_profile"
                if negated
                else "already_satisfied_by_query"
            )
    return None


def _format_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _format_price_range(price_range: Mapping[str, object], language: str) -> str:
    if "label" in price_range:
        label = str(price_range["label"])
        return f"tầm giá {label}" if language == "vi" else f"price range {label}"
    minimum = price_range.get("min_price")
    maximum = price_range.get("max_price")
    currency = str(price_range.get("currency", "UNKNOWN")).upper()

    def amount(value: object) -> str:
        number = _format_number(value)
        if currency == "USD":
            return f"${number}"
        if currency == "VND":
            return f"{number} VND"
        return number

    if minimum is not None and maximum is not None:
        if language == "vi":
            return f"giá từ {amount(minimum)} đến {amount(maximum)}"
        return f"priced from {amount(minimum)} to {amount(maximum)}"
    if maximum is not None:
        return f"giá dưới {amount(maximum)}" if language == "vi" else f"under {amount(maximum)}"
    return f"giá từ {amount(minimum)}" if language == "vi" else f"from {amount(minimum)}"


def _display_entity(field: str, value: object, language: str) -> str:
    raw = str(value).replace("_", " ")
    if language == "vi":
        if field == "brand":
            return f"thương hiệu {raw}"
        if field == "color":
            return f"màu {raw}"
        if field == "size":
            return f"size {raw}"
        if field == "material":
            return f"chất liệu {raw}"
        if field == "feature":
            return _VI_FEATURE_LABELS.get(str(value), raw)
        if field == "purpose":
            return f"phù hợp cho {_VI_PURPOSE_LABELS.get(str(value), raw)}"
        if field == "category":
            return raw
        return raw
    labels = {
        "brand": "brand",
        "color": "color",
        "size": "size",
        "material": "material",
    }
    prefix = labels.get(field)
    if field == "purpose":
        return f"suitable for {raw}"
    return f"{prefix} {raw}" if prefix else raw


def _display_negative_preference(field: str, value: object, language: str) -> str:
    display = _display_entity(field, value, language)
    # ``loại trừ`` avoids malformed Vietnamese such as "không không dây".
    # These clauses are an explainable trace; the router keeps them out of hard
    # entity filtering and applies the actual exclusion as a ranking penalty.
    return f"loại trừ {display}" if language == "vi" else f"exclude {display}"


class QueryRewriter:
    """Rewrite queries with intent templates and non-conflicting profile data."""

    def __init__(
        self,
        entity_extractor: EntityExtractor | None = None,
        max_added_preferences: int = 5,
    ) -> None:
        if max_added_preferences < 0:
            raise ValueError("max_added_preferences must be non-negative")
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.max_added_preferences = max_added_preferences

    def rewrite(
        self,
        original_query: str | None = None,
        intent: str | Mapping[str, Any] | None = None,
        entities: Mapping[str, Any] | None = None,
        user_profile: Mapping[str, Any] | None = None,
        *,
        query: str | None = None,
    ) -> QueryRewriteResult:
        """Return an explainable rewrite without overriding explicit constraints."""

        if query is not None and original_query is not None and query != original_query:
            raise ValueError("Pass either original_query or query, not two different values")
        query_text = query if query is not None else original_query
        query_text = query_text if isinstance(query_text, str) else ""
        result: QueryRewriteResult = {
            "original_query": query_text,
            "rewritten_query": query_text.strip(),
            "added_preferences": [],
            "ignored_preferences": [],
        }
        if not query_text.strip():
            return result

        normalized_intent = _normalize_intent(intent)
        extracted: Mapping[str, Any] = _entity_payload(entities)
        if not extracted:
            extracted = self.entity_extractor.extract(query_text)
        language = self.entity_extractor.detect_language(query_text)
        normalized_query = _normalized_value(query_text)
        additions = self._intent_additions(
            query_text, normalized_intent, extracted, language
        )

        if isinstance(user_profile, Mapping):
            profile = user_profile
        elif user_profile is not None and hasattr(user_profile, "to_dict"):
            raw_profile = user_profile.to_dict()
            profile = raw_profile if isinstance(raw_profile, Mapping) else {}
        else:
            profile = {}
        preferences = _collect_profile_preferences(profile)
        negative_preferences = _collect_negative_profile_preferences(profile)
        negative_markers = {
            (field, _normalized_value(value))
            for field, value in negative_preferences
        }
        price_range = _profile_price_range(profile)
        if price_range is not None:
            preferences.append(("price_range", price_range))

        if normalized_intent not in SUPPORTED_INTENTS:
            LOGGER.info("Unknown search intent %r; preserving the original query", intent)
            result["ignored_preferences"] = [
                {"field": field, "value": value, "reason": "unknown_intent"}
                for field, value in [*preferences, *negative_preferences]
            ]
            return result

        if normalized_intent not in PERSONALIZABLE_INTENTS:
            result["ignored_preferences"] = [
                {
                    "field": field,
                    "value": value,
                    "reason": "personalization_not_used_for_intent",
                }
                for field, value in [*preferences, *negative_preferences]
            ]
        else:
            for field, value in preferences:
                reason: str | None
                if (field, _normalized_value(value)) in negative_markers:
                    reason = "conflicts_with_negative_profile_preference"
                elif field == "price_range":
                    reason = (
                        "query_price_constraint_takes_priority"
                        if _entity_entries(extracted, "min_price")
                        or _entity_entries(extracted, "max_price")
                        else None
                    )
                else:
                    reason = _conflict_reason(field, value, extracted, normalized_query)

                if reason is not None:
                    result["ignored_preferences"].append(
                        {"field": field, "value": value, "reason": reason}
                    )
                    continue
                if len(result["added_preferences"]) >= self.max_added_preferences:
                    result["ignored_preferences"].append(
                        {"field": field, "value": value, "reason": "preference_limit"}
                    )
                    continue

                text = (
                    _format_price_range(value, language)
                    if field == "price_range" and isinstance(value, Mapping)
                    else _display_entity(field, value, language)
                )
                if _normalized_value(text) not in normalized_query:
                    additions.append(text)
                result["added_preferences"].append(
                    {"field": field, "value": value, "reason": "from_user_profile"}
                )

            for field, value in negative_preferences:
                reason = _negative_conflict_reason(field, value, extracted)
                if reason is not None:
                    result["ignored_preferences"].append(
                        {"field": field, "value": value, "reason": reason}
                    )
                    continue
                if len(result["added_preferences"]) >= self.max_added_preferences:
                    result["ignored_preferences"].append(
                        {"field": field, "value": value, "reason": "preference_limit"}
                    )
                    continue
                additions.append(_display_negative_preference(field, value, language))
                result["added_preferences"].append(
                    {
                        "field": field,
                        "value": value,
                        "reason": "from_user_profile_negative",
                    }
                )

        result["rewritten_query"] = self._combine(query_text.strip(), additions)
        LOGGER.debug(
            "Rewrote query for intent %s: %s", normalized_intent, result
        )
        return result

    def _intent_additions(
        self,
        query: str,
        intent: str,
        entities: Mapping[str, Any],
        language: str,
    ) -> list[str]:
        normalized_query = _normalized_value(query)
        additions: list[str] = []
        if intent == "need_based_search":
            for field in ("feature", "purpose"):
                for value, negated in _entity_values(entities, field):
                    if negated:
                        continue
                    display = _display_entity(field, value, language)
                    if _normalized_value(display) not in normalized_query:
                        additions.append(display)
        elif intent == "similar_product_search":
            markers = {"similar", "like", "tuong tu", "giong"}
            if not any(marker in normalized_query for marker in markers):
                additions.append(
                    "sản phẩm tương tự" if language == "vi" else "similar alternatives"
                )
        elif intent == "availability_check":
            markers = {"available", "availability", "con hang", "in stock", "stock"}
            if not any(marker in normalized_query for marker in markers):
                additions.append("còn hàng" if language == "vi" else "in stock")
        elif intent == "comparison":
            markers = {"compare", "comparison", "so sanh"}
            if not any(marker in normalized_query for marker in markers):
                additions.append(
                    "so sánh tính năng, giá và đánh giá"
                    if language == "vi"
                    else "compare features, price, and ratings"
                )
        return additions

    @staticmethod
    def _combine(query: str, additions: Iterable[str]) -> str:
        output = query.rstrip(" ;,")
        seen = {_normalized_value(output)}
        for addition in additions:
            clean = str(addition).strip(" ;,")
            marker = _normalized_value(clean)
            if not clean or marker in seen:
                continue
            output = f"{output}; {clean}" if output else clean
            seen.add(marker)
        return output


_DEFAULT_REWRITER = QueryRewriter()


def rewrite_query(
    original_query: str | None = None,
    intent: str | Mapping[str, Any] | None = None,
    entities: Mapping[str, Any] | None = None,
    user_profile: Mapping[str, Any] | None = None,
    *,
    query: str | None = None,
) -> QueryRewriteResult:
    """Convenience wrapper around :class:`QueryRewriter`."""

    return _DEFAULT_REWRITER.rewrite(
        original_query,
        intent=intent,
        entities=entities,
        user_profile=user_profile,
        query=query,
    )
