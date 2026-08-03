"""Strategy-based routing for product-discovery intents."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from src.nlp.entity_extractor import BRAND_ALIASES, normalize_text
from src.nlp.intent_taxonomy import SUPPORTED_INTENTS as SUPPORTED_SEARCH_INTENTS


class SearchEngine(Protocol):
    def search(self, query: str, top_k: int = 10, mode: str = "hybrid") -> list[dict]: ...


@dataclass
class SearchContext:
    query: str
    rewritten_query: str
    intent: str
    intent_confidence: float
    retrieval_query: str = ""
    entities: dict = field(default_factory=dict)
    user_id: str = ""
    user_profile: dict = field(default_factory=dict)
    product_id: str = ""
    mode: str = "hybrid"
    top_k: int = 5


class IntentStrategy:
    name = "product_search"

    def retrieve(self, router: "IntentRouter", context: SearchContext) -> list[dict]:
        # ``retrieval_query`` includes intent/rule expansion but deliberately
        # excludes profile clauses.  Profile-derived terms are soft ranking
        # evidence; sending them through HybridSearch would turn e.g. a preferred
        # brand into a hard catalog filter and can incorrectly empty the results.
        return router.search_engine.search(
            context.retrieval_query or context.query,
            top_k=max(context.top_k * 5, context.top_k),
            mode=context.mode,
        )


class ProductSearchStrategy(IntentStrategy):
    name = "product_search"


class NeedBasedSearchStrategy(IntentStrategy):
    name = "need_based_search"


class SimilarProductSearchStrategy(IntentStrategy):
    name = "similar_product_search"

    def retrieve(self, router: "IntentRouter", context: SearchContext) -> list[dict]:
        allowed_ids = _search_catalog_ids(router.search_engine)
        anchor_is_allowed = not allowed_ids or context.product_id in allowed_ids
        if context.product_id and anchor_is_allowed and router.recommender is not None:
            rows = router.recommender.similar_products(
                context.product_id, top_k=max(context.top_k * 5, context.top_k)
            )
            return [
                row
                | {
                    "semantic_score": float(row.get("score", 0.0)),
                    "lexical_score": float(row.get("score_breakdown", {}).get("Text", 0.0))
                    / 100.0,
                    "filter_reason": "item-to-item similarity",
                }
                for row in rows
                if (
                    str(row.get("product_id", "")) != context.product_id
                    and (
                        not allowed_ids
                        or str(row.get("product_id", "")) in allowed_ids
                    )
                )
            ]
        return super().retrieve(router, context)


class PersonalizedRecommendationStrategy(IntentStrategy):
    name = "personalized_recommendation"


class AvailabilityCheckStrategy(IntentStrategy):
    name = "availability_check"

    def retrieve(self, router: "IntentRouter", context: SearchContext) -> list[dict]:
        rows = super().retrieve(router, context)
        available = [row for row in rows if _explicitly_available(row)]
        for row in available:
            availability_reason = (
                "hard filter: demo inventory proxy above threshold"
                if row.get("availability_source")
                == "business_context_inventory_proxy"
                else "hard filter: explicit in-stock signal"
            )
            row["filter_reason"] = _append_reason(
                str(row.get("filter_reason", "")), availability_reason
            )
        return available


class ComparisonStrategy(IntentStrategy):
    name = "comparison"

    def retrieve(self, router: "IntentRouter", context: SearchContext) -> list[dict]:
        brand_anchors = _comparison_brand_anchors(context.query)
        if len(brand_anchors) >= 2:
            descriptor = _comparison_descriptor(context.query, brand_anchors)
            rows_by_anchor: list[tuple[str, list[dict]]] = []
            for brand in brand_anchors:
                anchor_query = " ".join(part for part in (brand, descriptor) if part)
                rows_by_anchor.append(
                    (
                        brand,
                        router.search_engine.search(
                            anchor_query,
                            top_k=max(context.top_k * 2, 4),
                            mode=context.mode,
                        ),
                    )
                )

            # Select one distinct candidate per anchor first, then fill any
            # remaining comparison slots.  This avoids a dominant first brand
            # crowding the other named alternative out of the result group.
            limit = max(2, min(context.top_k, 4))
            selected: list[dict] = []
            seen_product_ids: set[str] = set()
            for brand, rows in rows_by_anchor:
                for row in rows:
                    product_id = str(row.get("product_id", ""))
                    if not product_id or product_id in seen_product_ids:
                        continue
                    enriched = dict(row)
                    enriched["comparison_anchor"] = brand
                    selected.append(enriched)
                    seen_product_ids.add(product_id)
                    break
            for brand, rows in rows_by_anchor:
                for row in rows:
                    if len(selected) >= limit:
                        break
                    product_id = str(row.get("product_id", ""))
                    if not product_id or product_id in seen_product_ids:
                        continue
                    enriched = dict(row)
                    enriched["comparison_anchor"] = brand
                    selected.append(enriched)
                    seen_product_ids.add(product_id)
                if len(selected) >= limit:
                    break
            return selected[:limit]

        rows = super().retrieve(router, context)
        # A comparison should be a compact group, not a long recommendation feed.
        return rows[: max(2, min(context.top_k, 4))]


STRATEGY_TYPES = (
    ProductSearchStrategy,
    NeedBasedSearchStrategy,
    SimilarProductSearchStrategy,
    PersonalizedRecommendationStrategy,
    AvailabilityCheckStrategy,
    ComparisonStrategy,
)


SEEN_FILTER_INTENTS = {
    "personalized_recommendation",
    "need_based_search",
    # Compatibility for callers that used the earlier draft taxonomy label.
    "need_based_recommendation",
}


def _comparison_brand_anchors(query: str) -> list[str]:
    normalized = normalize_text(query)
    matches: list[tuple[int, str]] = []
    for canonical, aliases in BRAND_ALIASES.items():
        candidates = {canonical, *aliases}
        starts = [
            match.start()
            for candidate in candidates
            if (
                match := re.search(
                    rf"(?<![a-z0-9]){re.escape(normalize_text(candidate))}(?![a-z0-9])",
                    normalized,
                )
            )
        ]
        if starts:
            matches.append((min(starts), canonical))
    return [canonical for _, canonical in sorted(matches)]


def _comparison_descriptor(query: str, anchors: list[str]) -> str:
    descriptor = normalize_text(query)
    for canonical in anchors:
        for alias in {canonical, *BRAND_ALIASES.get(canonical, set())}:
            descriptor = re.sub(
                rf"(?<![a-z0-9]){re.escape(normalize_text(alias))}(?![a-z0-9])",
                " ",
                descriptor,
            )
    descriptor = re.sub(
        r"\b(?:compare|comparison|versus|vs|and|so sanh|voi|va)\b",
        " ",
        descriptor,
    )
    return " ".join(descriptor.split())


class IntentRouter:
    """Detect a query intent and dispatch it through an extensible strategy."""

    def __init__(
        self,
        search_engine: SearchEngine,
        *,
        intent_classifier: object | None = None,
        entity_extractor: object | None = None,
        query_rewriter: object | None = None,
        recommender: object | None = None,
        strategies: list[IntentStrategy] | None = None,
    ) -> None:
        self.search_engine = search_engine
        self.intent_classifier = intent_classifier
        self.entity_extractor = entity_extractor
        self.query_rewriter = query_rewriter
        self.recommender = recommender
        instances = strategies or [strategy_type() for strategy_type in STRATEGY_TYPES]
        self.strategies = {strategy.name: strategy for strategy in instances}

    def route(
        self,
        query: str,
        *,
        user_id: str = "",
        user_profile: dict | None = None,
        product_id: str = "",
        top_k: int = 5,
        mode: str = "hybrid",
        intent: str | None = None,
    ) -> dict:
        clean_query = str(query).strip()
        if not clean_query:
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        detected_intent, confidence = self._detect_intent(clean_query, intent)
        if detected_intent == "comparison" and top_k < 2:
            raise ValueError("comparison intent requires top_k >= 2")
        entities = self._extract_entities(clean_query)
        profile = user_profile or {}
        rewrite = self._rewrite(clean_query, detected_intent, entities, profile)
        rewritten_query = str(rewrite.get("rewritten_query") or clean_query)
        # Build a second, profile-free form for retrieval.  This preserves the
        # measurable effect of intent-aware query rewriting while maintaining
        # provenance: only constraints stated in the user's query can become
        # hard entity filters; historical preferences stay in the ranker.
        retrieval_rewrite = (
            self._rewrite(clean_query, detected_intent, entities, {})
            if profile
            else rewrite
        )
        retrieval_query = str(
            retrieval_rewrite.get("rewritten_query") or clean_query
        )
        source_product_id = self._resolve_source_product_id(
            clean_query,
            detected_intent,
            product_id,
        )
        context = SearchContext(
            query=clean_query,
            rewritten_query=rewritten_query,
            retrieval_query=retrieval_query,
            intent=detected_intent,
            intent_confidence=confidence,
            entities=entities,
            user_id=user_id,
            user_profile=profile,
            product_id=source_product_id,
            mode=mode,
            top_k=top_k,
        )
        strategy = self.strategies.get(detected_intent, self.strategies["product_search"])
        candidates = strategy.retrieve(self, context)
        seen_filtered_count = 0
        if (
            detected_intent in SEEN_FILTER_INTENTS
            and user_id
            and self.recommender is not None
            and hasattr(self.recommender, "seen_by_user")
        ):
            seen = self.recommender.seen_by_user.get(user_id, set())
            before = len(candidates)
            candidates = [
                row for row in candidates if str(row.get("product_id", "")) not in seen
            ]
            seen_filtered_count = before - len(candidates)
        result_top_k = top_k
        if self.recommender is not None and hasattr(
            self.recommender, "rank_intent_candidates"
        ):
            results = self.recommender.rank_intent_candidates(
                candidates,
                intent=detected_intent,
                user_id=user_id,
                extracted_entities=entities,
                user_profile=profile,
                top_k=result_top_k,
            )
        else:
            results = sorted(
                candidates,
                key=lambda row: float(row.get("final_score", row.get("score", 0.0))),
                reverse=True,
            )[:result_top_k]
        return {
            "original_query": clean_query,
            "detected_intent": detected_intent,
            "intent_confidence": round(confidence, 6),
            "extracted_entities": entities,
            "rewritten_query": rewritten_query,
            "retrieval_query": retrieval_query,
            "added_preferences": rewrite.get("added_preferences", []),
            "ignored_preferences": rewrite.get("ignored_preferences", []),
            "search_mode": mode,
            "strategy": strategy.name,
            "source_product_id": source_product_id,
            "candidate_count": len(candidates),
            "seen_filtered_count": seen_filtered_count,
            "results": results,
        }

    def _detect_intent(self, query: str, override: str | None) -> tuple[str, float]:
        if override:
            return (
                override if override in self.strategies else "product_search",
                1.0 if override in self.strategies else 0.0,
            )
        if self.intent_classifier is None:
            return "product_search", 0.0
        raw = self.intent_classifier.predict(query)
        if isinstance(raw, str):
            return (raw if raw in self.strategies else "product_search", 1.0)
        data = _as_dict(raw)
        label = str(data.get("intent", "product_search"))
        return (
            label if label in self.strategies else "product_search",
            float(data.get("confidence", 0.0)),
        )

    def _extract_entities(self, query: str) -> dict:
        if self.entity_extractor is None:
            return {}
        raw = self.entity_extractor.extract(query)
        data = _as_dict(raw)
        nested = data.get("entities")
        return dict(nested) if isinstance(nested, dict) else data

    def _rewrite(self, query: str, intent: str, entities: dict, profile: dict) -> dict:
        if self.query_rewriter is None:
            return {
                "original_query": query,
                "rewritten_query": query,
                "added_preferences": [],
                "ignored_preferences": [],
            }
        raw = self.query_rewriter.rewrite(query, intent, entities, profile)
        return _as_dict(raw)

    def _resolve_source_product_id(
        self,
        query: str,
        intent: str,
        explicit_product_id: str,
    ) -> str:
        """Resolve a similar-item anchor from an explicit ID or exact catalog title."""

        allowed_ids = _search_catalog_ids(self.search_engine)
        if explicit_product_id:
            resolved = str(explicit_product_id)
            return resolved if not allowed_ids or resolved in allowed_ids else ""
        if intent != "similar_product_search" or self.recommender is None:
            return ""
        raw_products = getattr(self.recommender, "products", {})
        if isinstance(raw_products, dict):
            product_rows = [
                (str(product_id), row)
                for product_id, row in raw_products.items()
                if isinstance(row, dict)
            ]
        else:
            product_rows = [
                (str(row.get("product_id", "")), row)
                for row in raw_products or []
                if isinstance(row, dict) and row.get("product_id")
            ]
        normalized_query = " ".join(normalize_text(query).split())
        candidates: list[tuple[int, int, str]] = []
        for product_id, product in product_rows:
            if allowed_ids and product_id not in allowed_ids:
                continue
            normalized_id = normalize_text(product_id).strip()
            if normalized_id and re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_id)}(?![a-z0-9])",
                normalized_query,
            ):
                candidates.append((2, len(normalized_id), product_id))
                continue
            title = " ".join(normalize_text(product.get("title", "")).split())
            # Very short titles are unsafe substring anchors; generated/eval
            # queries use full product titles and therefore satisfy this guard.
            if len(title) >= 4 and title in normalized_query:
                candidates.append((1, len(title), product_id))
        if not candidates:
            return ""
        candidates.sort(key=lambda row: (-row[0], -row[1], row[2]))
        return candidates[0][2]


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    try:
        return asdict(value)
    except (TypeError, ValueError):
        return {}


def _search_catalog_ids(search_engine: object) -> set[str]:
    raw_products = getattr(search_engine, "products", None)
    if isinstance(raw_products, dict):
        return {str(product_id) for product_id in raw_products}
    raw_ids = getattr(search_engine, "product_ids", None)
    if raw_ids is not None:
        return {str(product_id) for product_id in raw_ids}
    return set()


def _explicitly_available(product: dict) -> bool:
    raw = product.get("in_stock", product.get("available", product.get("availability")))
    return str(raw or "").strip().casefold() in {
        "1", "true", "yes", "available", "in stock", "còn hàng", "con hang"
    }


def _append_reason(existing: str, reason: str) -> str:
    return f"{existing}; {reason}" if existing else reason
