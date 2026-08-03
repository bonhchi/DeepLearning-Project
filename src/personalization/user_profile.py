"""Leakage-safe user profiles for intent-aware product search."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from typing import Iterable

from src.io_utils import json_loads_safe, parse_float, parse_int


@dataclass
class UserProfile:
    """Compact preferences derived exclusively from training interactions."""

    user_id: str
    preferred_categories: list[str] = field(default_factory=list)
    brands: list[str] = field(default_factory=list)
    price_range: dict[str, float] = field(default_factory=dict)
    attributes: list[str] = field(default_factory=list)
    positive_preferences: list[str] = field(default_factory=list)
    negative_preferences: list[str] = field(default_factory=list)
    interaction_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _timestamp_seconds(value: object) -> float:
    raw = parse_float(value, 0.0)
    if raw > 10_000_000_000:
        return raw / 1000.0
    return raw


def _event_timestamp_key(value: object) -> str:
    """Canonicalize a raw event timestamp without dropping millisecond precision."""

    raw = str(value or "").strip()
    try:
        parsed = Decimal(raw)
    except (InvalidOperation, ValueError):
        return raw
    if not parsed.is_finite():
        return raw
    return format(parsed.normalize(), "f")


def _feature_tokens(product: dict) -> list[str]:
    raw = json_loads_safe(product.get("features"), default=[])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    output: list[str] = []
    for feature in raw:
        text = str(feature).strip().casefold()
        if text and len(text) <= 80:
            output.append(text)
    return output


class UserProfileBuilder:
    """Aggregate weighted preferences with exponential time decay.

    Validation and test rows are always ignored so a profile cannot learn from the
    same held-out items used by retrieval evaluation.
    """

    def __init__(self, half_life_days: float = 180.0, top_n: int = 5) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        self.half_life_days = float(half_life_days)
        self.top_n = int(top_n)

    def build(
        self,
        user_id: str,
        interactions: Iterable[dict],
        products: Iterable[dict],
        reviews: Iterable[dict] | None = None,
        reference_timestamp: float | None = None,
    ) -> UserProfile:
        product_by_id = {
            str(row.get("product_id", "")): row
            for row in products
            if row.get("product_id")
        }
        rows = [
            row
            for row in interactions
            if str(row.get("user_id", "")) == user_id
            and row.get("split", "train") == "train"
        ]
        if not rows:
            return UserProfile(user_id=user_id)

        timestamps = [_timestamp_seconds(row.get("timestamp")) for row in rows]
        now = reference_timestamp or max(timestamps, default=0.0)
        if now <= 0:
            now = datetime.now(timezone.utc).timestamp()

        category_scores: dict[str, float] = defaultdict(float)
        brand_scores: dict[str, float] = defaultdict(float)
        attribute_scores: dict[str, float] = defaultdict(float)
        negative_scores: dict[str, float] = defaultdict(float)
        review_term_scores: dict[str, float] = defaultdict(float)
        weighted_prices: list[tuple[float, float]] = []

        for row in rows:
            product = product_by_id.get(str(row.get("product_id", "")), {})
            timestamp = _timestamp_seconds(row.get("timestamp"))
            age_days = max(now - timestamp, 0.0) / 86_400.0 if timestamp else 0.0
            decay = math.pow(0.5, age_days / self.half_life_days)
            positive = parse_int(row.get("label"), 0) == 1 or parse_float(row.get("rating"), 0.0) >= 4.0
            strength = max(parse_float(row.get("event_weight"), 1.0), 0.1) * decay
            signed_strength = strength if positive else -strength

            category = str(product.get("category", "")).strip()
            brand = str(product.get("brand") or product.get("store") or "").strip()
            if category:
                category_scores[category] += signed_strength
                if not positive:
                    negative_scores[f"category:{category}"] += strength
            if brand:
                brand_scores[brand] += signed_strength
                if not positive:
                    negative_scores[f"brand:{brand}"] += strength
            for feature in _feature_tokens(product):
                attribute_scores[feature] += signed_strength
                if not positive:
                    negative_scores[f"attribute:{feature}"] += strength
            price = parse_float(product.get("price"), 0.0)
            if positive and price > 0:
                weighted_prices.append((price, strength))

        preferred_categories = self._top_positive(category_scores)
        brands = self._top_positive(brand_scores)
        attributes = self._top_positive(attribute_scores)
        positives = [
            *[f"category:{value}" for value in preferred_categories],
            *[f"brand:{value}" for value in brands],
            *[f"attribute:{value}" for value in attributes],
        ]
        negatives = [
            key
            for key, _ in sorted(negative_scores.items(), key=lambda item: (-item[1], item[0]))
        ][: self.top_n]

        price_range: dict[str, float] = {}
        if weighted_prices:
            total_weight = sum(weight for _, weight in weighted_prices)
            average = sum(price * weight for price, weight in weighted_prices) / total_weight
            prices = sorted(price for price, _ in weighted_prices)
            price_range = {
                "min": round(prices[0], 2),
                "max": round(prices[-1], 2),
                "average": round(average, 2),
            }

        if reviews is not None:
            train_review_keys = {
                (
                    str(row.get("product_id", "")),
                    _event_timestamp_key(row.get("timestamp")),
                )
                for row in rows
            }
            for review in reviews:
                if str(review.get("user_id", "")) != user_id:
                    continue
                # reviews.csv has no split column, so join the exact product-event
                # timestamp to its interaction split. Product id alone is unsafe
                # when one user reviews the same item more than once.
                review_key = (
                    str(review.get("product_id", "")),
                    _event_timestamp_key(review.get("timestamp")),
                )
                if review_key not in train_review_keys:
                    continue
                rating = parse_float(review.get("rating"), 0.0)
                signed = 1.0 if rating >= 4.0 else -1.0 if rating and rating <= 2.0 else 0.0
                if not signed:
                    continue
                review_timestamp = _timestamp_seconds(review.get("timestamp"))
                review_age_days = (
                    max(now - review_timestamp, 0.0) / 86_400.0
                    if review_timestamp
                    else 0.0
                )
                review_decay = math.pow(0.5, review_age_days / self.half_life_days)
                text = f"{review.get('review_title', '')} {review.get('review_text', '')}"
                tokens = [
                    token
                    for token in re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE)
                    if len(token) >= 4
                ]
                for token, count in Counter(tokens).most_common(12):
                    review_term_scores[token] += signed * min(count, 3) * review_decay

        positive_review_terms = [
            term
            for term, score in sorted(
                review_term_scores.items(), key=lambda item: (-item[1], item[0])
            )
            if score > 0
        ][: self.top_n]
        negative_review_terms = [
            term
            for term, score in sorted(
                review_term_scores.items(), key=lambda item: (item[1], item[0])
            )
            if score < 0
        ][: self.top_n]
        positives.extend(f"review_term:{term}" for term in positive_review_terms)
        negatives.extend(f"review_term:{term}" for term in negative_review_terms)

        return UserProfile(
            user_id=user_id,
            preferred_categories=preferred_categories,
            brands=brands,
            price_range=price_range,
            attributes=attributes,
            positive_preferences=positives,
            negative_preferences=list(dict.fromkeys(negatives))[: self.top_n],
            interaction_count=len(rows),
        )

    def build_all(
        self,
        interactions: Iterable[dict],
        products: Iterable[dict],
        reviews: Iterable[dict] | None = None,
    ) -> dict[str, UserProfile]:
        interaction_rows = list(interactions)
        product_rows = list(products)
        review_rows = list(reviews or [])
        user_ids = sorted(
            {
                str(row.get("user_id"))
                for row in interaction_rows
                if row.get("user_id") and row.get("split", "train") == "train"
            }
        )
        return {
            user_id: self.build(user_id, interaction_rows, product_rows, review_rows)
            for user_id in user_ids
        }

    def _top_positive(self, scores: dict[str, float]) -> list[str]:
        return [
            key
            for key, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            if score > 0
        ][: self.top_n]


def serialize_profiles(profiles: dict[str, UserProfile]) -> str:
    """Serialize profiles without exposing raw interaction histories."""

    return json.dumps(
        {user_id: profile.to_dict() for user_id, profile in profiles.items()},
        ensure_ascii=False,
        indent=2,
    )
