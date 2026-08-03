"""Evidence-based explanations for intent-conditioned rankings."""

from __future__ import annotations

from collections.abc import Mapping


SCORE_LABELS = {
    "semantic_score": "nội dung gần với truy vấn",
    "lexical_score": "khớp trực tiếp từ khóa",
    "entity_match_score": "đáp ứng các điều kiện đã nêu",
    "user_preference_score": "phù hợp lịch sử mua sắm",
    "quality_score": "có đánh giá tốt",
    "popularity_score": "được nhiều người quan tâm",
    "availability_score": "đang có sẵn",
}


class ExplanationBuilder:
    def build(
        self,
        *,
        product: Mapping[str, object],
        intent: str,
        score_breakdown: Mapping[str, float],
        matched_entities: Mapping[str, object] | None = None,
        user_profile: Mapping[str, object] | None = None,
    ) -> str:
        reasons: list[str] = []
        entities = matched_entities or {}
        for name in ("feature", "purpose", "brand", "color", "material", "size", "category"):
            raw = entities.get(name)
            value = raw.get("value") if isinstance(raw, Mapping) else raw
            if value not in (None, "", []):
                reasons.append(f"khớp {name} {self._display(value)}")
        price = self._price_reason(product, entities)
        if price:
            reasons.append(price)

        ranked_components = sorted(
            (
                (name, float(value))
                for name, value in score_breakdown.items()
                if name in SCORE_LABELS and float(value) > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        for name, _ in ranked_components:
            label = SCORE_LABELS[name]
            if label not in reasons:
                reasons.append(label)
            if len(reasons) >= 3:
                break

        profile = user_profile or {}
        brand = str(product.get("brand") or product.get("store") or "")
        if brand and brand in set(profile.get("brands", []) or []):
            reasons.append(f"gần với thương hiệu bạn thường xem ({brand})")
        category = str(product.get("category", ""))
        if category and category in set(profile.get("preferred_categories", []) or []):
            reasons.append(f"thuộc nhóm bạn thường quan tâm ({category})")

        if intent == "comparison":
            prefix = "Được đưa vào nhóm so sánh vì "
        elif intent == "availability_check":
            prefix = "Kết quả kiểm tra phù hợp vì "
        else:
            prefix = "Phù hợp vì "
        if not reasons:
            title = str(product.get("title") or product.get("product_id") or "sản phẩm này")
            return f"{title} được giữ lại theo chiến lược {intent}."
        return prefix + ", ".join(list(dict.fromkeys(reasons))[:3]) + "."

    @staticmethod
    def _display(value: object) -> str:
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value)
        return str(value)

    @staticmethod
    def _price_reason(product: Mapping[str, object], entities: Mapping[str, object]) -> str:
        try:
            price = float(product.get("price") or 0.0)
        except (TypeError, ValueError):
            return ""
        if price <= 0:
            return ""
        minimum = entities.get("min_price")
        maximum = entities.get("max_price")
        if isinstance(minimum, Mapping):
            minimum = minimum.get("value")
        if isinstance(maximum, Mapping):
            maximum = maximum.get("value")
        try:
            if maximum is not None and price <= float(maximum):
                return "nằm dưới mức giá yêu cầu"
            if minimum is not None and price >= float(minimum):
                return "nằm trên mức giá tối thiểu"
        except (TypeError, ValueError):
            return ""
        return ""
