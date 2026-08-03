"""Intent-aware lexical, semantic, and hybrid product retrieval.

This module owns retrieval only.  Intent classification, entity extraction,
text encoding, and metadata filtering are injected dependencies so later NLP
and personalization modules can evolve without coupling to this pipeline.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from src.feature_extraction.embeddings import (
    DenseTextEncoder,
    TextEncoder,
    TfidfEncoder,
    product_text,
)
from src.io_utils import json_loads_safe, parse_float
from src.semantic_search.lexical_index import SparseTfidfIndex
from src.semantic_search.vector_index import VectorIndex


LOGGER = logging.getLogger(__name__)
SEARCH_MODES = frozenset({"lexical", "semantic", "hybrid"})
UNICODE_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class HybridSearchEngine:
    """Composable product retrieval pipeline.

    The classifier dependency may expose ``predict(query)`` or be callable.
    The entity extractor may expose ``extract(query)``, ``extract_entities``,
    or be callable.  Their outputs can be dictionaries, dataclasses/objects,
    strings (intent only), or ``(value, confidence)`` tuples.
    """

    def __init__(
        self,
        products: Sequence[Mapping[str, Any]],
        vector_index: VectorIndex | None = None,
        lexical_encoder: TextEncoder | None = None,
        dense_encoder: TextEncoder | None = None,
        intent_classifier: Any | None = None,
        entity_extractor: Any | None = None,
        metadata_filter: Callable[[Mapping[str, Any], Mapping[str, Any]], Any] | None = None,
        semantic_weight: float = 0.65,
        lexical_weight: float = 0.35,
        candidate_pool_size: int = 200,
        strict_catalog: bool = True,
        currency_rates_to_catalog: Mapping[str, float] | None = None,
        text_builder: Callable[[dict[str, Any]], str] = product_text,
    ) -> None:
        if not products:
            raise ValueError("HybridSearchEngine requires at least one product")
        if semantic_weight < 0.0 or lexical_weight < 0.0:
            raise ValueError("Search weights must be non-negative")
        if semantic_weight + lexical_weight <= 0.0:
            raise ValueError("At least one search weight must be positive")
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be greater than zero")

        self.products: dict[str, dict[str, Any]] = {}
        self.product_ids: list[str] = []
        for position, raw_product in enumerate(products):
            product = dict(raw_product)
            product_id = str(product.get("product_id", "")).strip()
            if not product_id:
                raise ValueError(f"Missing product_id at catalog position {position}")
            if product_id in self.products:
                raise ValueError(f"Duplicate product_id in catalog: {product_id}")
            product["product_id"] = product_id
            self.products[product_id] = product
            self.product_ids.append(product_id)

        self.vector_index = vector_index
        self.lexical_index: SparseTfidfIndex | VectorIndex | None = None
        self.lexical_encoder = lexical_encoder or TfidfEncoder(max_features=2048)
        self.dense_encoder = dense_encoder or DenseTextEncoder()
        self.intent_classifier = intent_classifier
        self.entity_extractor = entity_extractor
        self.metadata_filter = metadata_filter
        total_weight = semantic_weight + lexical_weight
        self.semantic_weight = semantic_weight / total_weight
        self.lexical_weight = lexical_weight / total_weight
        self.candidate_pool_size = candidate_pool_size
        self.strict_catalog = strict_catalog
        self.currency_rates_to_catalog = {
            str(currency).upper(): float(rate)
            for currency, rate in (currency_rates_to_catalog or {"USD": 1.0}).items()
        }
        if any(rate <= 0.0 for rate in self.currency_rates_to_catalog.values()):
            raise ValueError("Currency conversion rates must be positive")
        self.text_builder = text_builder
        self.last_trace: dict[str, Any] = {}

        if self.vector_index is not None:
            self._validate_index_catalog(self.vector_index, "semantic")

    def build_indices(
        self,
        build_lexical: bool = True,
        build_semantic: bool = True,
    ) -> "HybridSearchEngine":
        """Build any non-injected retrieval indices from the current catalog."""

        documents = [self.text_builder(self.products[product_id]) for product_id in self.product_ids]
        if build_lexical and self.lexical_index is None:
            if getattr(self.lexical_encoder, "dimension", 0) == 0 or not getattr(
                self.lexical_encoder, "is_fitted", True
            ):
                self.lexical_encoder.fit(documents)
            if isinstance(self.lexical_encoder, TfidfEncoder):
                self.lexical_index = SparseTfidfIndex().build(
                    self.product_ids,
                    documents,
                    self.lexical_encoder,
                )
            else:
                lexical_vectors = self._encode_batch(self.lexical_encoder, documents)
                self.lexical_index = VectorIndex(use_faiss=False).build(
                    self.product_ids,
                    lexical_vectors,
                )

        if build_semantic and self.vector_index is None:
            semantic_vectors = self._encode_batch(self.dense_encoder, documents)
            self.vector_index = VectorIndex().build(self.product_ids, semantic_vectors)
        return self

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Retrieve and filter products for one query.

        Every result contains ``semantic_score``, ``lexical_score``,
        ``matched_entities``, and ``filter_reason`` in addition to product
        metadata and the detected intent fields.
        """

        clean_query = str(query).strip()
        if not clean_query:
            raise ValueError("query must not be empty")
        normalized_mode = mode.strip().lower()
        if normalized_mode not in SEARCH_MODES:
            raise ValueError(
                f"Unsupported search mode {mode!r}; expected lexical, semantic, or hybrid"
            )
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        intent, intent_confidence = self._detect_intent(clean_query)
        entities = self._extract_entities(clean_query)
        pool_size = min(
            len(self.product_ids),
            max(self.candidate_pool_size, top_k * 10),
        )

        lexical_query: list[float] | None = None
        semantic_query: list[float] | None = None
        if normalized_mode in {"lexical", "hybrid"}:
            self.build_indices(build_lexical=True, build_semantic=False)
            if self.lexical_index is None:  # Defensive guard for custom overrides.
                raise RuntimeError("Lexical index is unavailable")
            lexical_query = self._encode_one(self.lexical_encoder, clean_query)

        if normalized_mode in {"semantic", "hybrid"}:
            self.build_indices(build_lexical=False, build_semantic=True)
            if self.vector_index is None:
                raise RuntimeError("Semantic index is unavailable")
            semantic_query = self._encode_one(self.dense_encoder, clean_query)

        adaptive_rounds = 0
        while True:
            lexical_scores: dict[str, float] = {}
            semantic_scores: dict[str, float] = {}
            if lexical_query is not None and self.lexical_index is not None:
                lexical_scores = {
                    result["product_id"]: float(result["score"])
                    for result in self.lexical_index.search(lexical_query, pool_size)
                }
            if semantic_query is not None and self.vector_index is not None:
                semantic_scores = {
                    result["product_id"]: float(result["score"])
                    for result in self.vector_index.search(semantic_query, pool_size)
                }

            candidate_ids = set(lexical_scores) | set(semantic_scores)
            rows = []
            filtered_count = 0
            filter_reasons: Counter[str] = Counter()
            for product_id in candidate_ids:
                product = self.products.get(product_id)
                if product is None:
                    if self.strict_catalog:
                        raise ValueError(
                            f"Index/catalog mismatch: product {product_id!r} is not in the catalog"
                        )
                    continue
                passed, matched_entities, filter_reason = self._apply_metadata_filter(
                    product,
                    entities,
                )
                filter_reasons[filter_reason] += 1
                if not passed:
                    filtered_count += 1
                    continue

                semantic_score = semantic_scores.get(product_id, 0.0)
                lexical_score = lexical_scores.get(product_id, 0.0)
                if normalized_mode == "lexical":
                    final_score = lexical_score
                elif normalized_mode == "semantic":
                    final_score = semantic_score
                else:
                    final_score = (
                        self.semantic_weight * semantic_score
                        + self.lexical_weight * lexical_score
                    )
                row = dict(product)
                row.update(
                    {
                        "intent": intent,
                        "intent_confidence": intent_confidence,
                        "semantic_score": semantic_score,
                        "lexical_score": lexical_score,
                        "matched_entities": matched_entities,
                        "filter_reason": filter_reason,
                        "final_score": final_score,
                        "search_mode": normalized_mode,
                    }
                )
                rows.append(row)

            if (
                len(rows) >= top_k
                or not filtered_count
                or pool_size >= len(self.product_ids)
            ):
                break
            pool_size = min(len(self.product_ids), max(pool_size + 1, pool_size * 2))
            adaptive_rounds += 1

        rows.sort(
            key=lambda row: (-float(row["final_score"]), str(row["product_id"]))
        )
        results = rows[:top_k]
        self.last_trace = {
            "query": clean_query,
            "mode": normalized_mode,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "extracted_entities": entities,
            "candidate_count": len(candidate_ids),
            "retrieval_pool_size": pool_size,
            "adaptive_overfetch_rounds": adaptive_rounds,
            "filtered_count": filtered_count,
            "applied_filters": dict(sorted(filter_reasons.items())),
            "returned_count": len(results),
        }
        return results

    def _validate_index_catalog(self, index: VectorIndex, label: str) -> None:
        if not index.is_built:
            raise ValueError(f"Injected {label} index has not been built")
        catalog_ids = set(self.product_ids)
        index_ids = set(index.product_ids)
        missing = sorted(catalog_ids - index_ids)
        unknown = sorted(index_ids - catalog_ids)
        if self.strict_catalog and (missing or unknown):
            raise ValueError(
                f"{label.capitalize()} index/catalog mismatch: "
                f"missing={missing[:5]}, unknown={unknown[:5]}"
            )

    @staticmethod
    def _encode_batch(encoder: TextEncoder, texts: list[str]) -> list[list[float]]:
        encoded = encoder.encode(texts)
        if not isinstance(encoded, list):
            raise TypeError("Text encoder must return a list for batch input")
        if encoded and isinstance(encoded[0], (int, float)):
            raise ValueError("Text encoder returned one vector for a batch")
        return [[float(value) for value in vector] for vector in encoded]  # type: ignore[union-attr]

    @staticmethod
    def _encode_one(encoder: TextEncoder, text: str) -> list[float]:
        if hasattr(encoder, "encode_one"):
            return [float(value) for value in encoder.encode_one(text)]
        encoded = encoder.encode(text)
        if encoded and isinstance(encoded[0], list):  # type: ignore[index]
            encoded = encoded[0]  # type: ignore[index,assignment]
        return [float(value) for value in encoded]  # type: ignore[union-attr]

    def _detect_intent(self, query: str) -> tuple[str, float]:
        if self.intent_classifier is None:
            return "unknown", 0.0
        detector = self.intent_classifier
        if hasattr(detector, "predict"):
            raw = detector.predict(query)
        elif callable(detector):
            raw = detector(query)
        else:
            raise TypeError("intent_classifier must be callable or expose predict(query)")

        if isinstance(raw, list) and len(raw) == 1:
            raw = raw[0]
        if isinstance(raw, str):
            return raw, 1.0
        if isinstance(raw, tuple) and raw:
            confidence = float(raw[1]) if len(raw) > 1 else 1.0
            return str(raw[0]), confidence
        if isinstance(raw, Mapping):
            intent = raw.get("intent", raw.get("label", raw.get("predicted_intent", "unknown")))
            confidence = raw.get("confidence", raw.get("probability", raw.get("score", 0.0)))
            return str(intent), float(confidence)
        intent = getattr(raw, "intent", getattr(raw, "label", "unknown"))
        confidence = getattr(raw, "confidence", getattr(raw, "probability", 0.0))
        return str(intent), float(confidence)

    def _extract_entities(self, query: str) -> dict[str, Any]:
        if self.entity_extractor is None:
            return {}
        extractor = self.entity_extractor
        if hasattr(extractor, "extract"):
            raw = extractor.extract(query)
        elif hasattr(extractor, "extract_entities"):
            raw = extractor.extract_entities(query)
        elif callable(extractor):
            raw = extractor(query)
        else:
            raise TypeError("entity_extractor must be callable or expose extract(query)")

        if raw is None:
            return {}
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        elif not isinstance(raw, (Mapping, list)) and hasattr(raw, "__dict__"):
            raw = vars(raw)
        if isinstance(raw, Mapping):
            nested = raw.get("entities")
            return dict(nested) if isinstance(nested, Mapping) else dict(raw)
        if isinstance(raw, list):
            entities: dict[str, Any] = {}
            for item in raw:
                if not isinstance(item, Mapping):
                    continue
                name = item.get("type", item.get("name", item.get("entity")))
                if name:
                    entities[str(name)] = item
            return entities
        raise TypeError("Entity extractor output must be a mapping or list")

    def _apply_metadata_filter(
        self,
        product: Mapping[str, Any],
        entities: Mapping[str, Any],
    ) -> tuple[bool, dict[str, Any], str]:
        if self.metadata_filter is not None:
            return self._normalize_filter_result(self.metadata_filter(product, entities))
        return self._default_metadata_filter(product, entities)

    @staticmethod
    def _normalize_filter_result(raw: Any) -> tuple[bool, dict[str, Any], str]:
        if isinstance(raw, bool):
            return raw, {}, "custom_filter_passed" if raw else "custom_filter_rejected"
        if isinstance(raw, tuple):
            passed = bool(raw[0]) if raw else False
            matched = dict(raw[1]) if len(raw) > 1 and isinstance(raw[1], Mapping) else {}
            reason = str(raw[2]) if len(raw) > 2 else (
                "custom_filter_passed" if passed else "custom_filter_rejected"
            )
            return passed, matched, reason
        if isinstance(raw, Mapping):
            passed = bool(raw.get("passed", raw.get("include", False)))
            matched = raw.get("matched_entities", {})
            reason = raw.get("filter_reason", raw.get("reason", "custom_filter"))
            return passed, dict(matched) if isinstance(matched, Mapping) else {}, str(reason)
        raise TypeError("metadata_filter must return bool, tuple, or mapping")

    def _default_metadata_filter(
        self,
        product: Mapping[str, Any],
        entities: Mapping[str, Any],
    ) -> tuple[bool, dict[str, Any], str]:
        normalized_entities = {
            str(name).lower(): value
            for name, value in entities.items()
            if self._entity_value(value) not in (None, "", [], {})
        }
        if not normalized_entities:
            return True, {}, "no_metadata_filters"

        matched: dict[str, Any] = {}
        reasons: list[str] = []
        product_price = parse_float(product.get("price"), -1.0)
        min_price, min_currency_error = self._normalized_price_constraint(
            normalized_entities.get("min_price")
        )
        max_price, max_currency_error = self._normalized_price_constraint(
            normalized_entities.get("max_price")
        )
        currency_error = min_currency_error or max_currency_error
        if currency_error:
            return False, matched, currency_error
        if min_price is not None:
            if product_price < 0.0:
                return False, matched, "rejected: missing price"
            if product_price < min_price:
                return False, matched, f"rejected: price below {min_price:g}"
            matched["min_price"] = product_price
            reasons.append(f"min_price={min_price:g}")
        if max_price is not None:
            if product_price < 0.0:
                return False, matched, "rejected: missing price"
            if product_price > max_price:
                return False, matched, f"rejected: price above {max_price:g}"
            matched["max_price"] = product_price
            reasons.append(f"max_price={max_price:g}")

        aliases = {
            "brand": ("brand", "store"),
            "category": ("category",),
            "color": ("color", "features", "description", "title"),
            "size": ("size", "features", "description", "title"),
            "material": ("material", "features", "description", "title"),
            "feature": ("features", "description", "title"),
            "features": ("features", "description", "title"),
            "purpose": ("description", "features", "title", "category"),
        }
        for entity_name, product_fields in aliases.items():
            if entity_name not in normalized_entities:
                continue
            expected_values, excluded_values = self._constraint_values(
                normalized_entities[entity_name]
            )
            actual_values = [
                self._metadata_text(product.get(field))
                for field in product_fields
                if product.get(field) not in (None, "")
            ]
            actual_values = [value for value in actual_values if value]
            if expected_values and not actual_values:
                return False, matched, f"rejected: missing {entity_name} metadata"
            expected_matches = [
                expected
                for expected in expected_values
                if any(self._text_matches(expected, actual) for actual in actual_values)
            ]
            if len(expected_matches) != len(expected_values):
                return False, matched, f"rejected: {entity_name} mismatch"
            forbidden_matches = [
                excluded
                for excluded in excluded_values
                if any(self._text_matches(excluded, actual) for actual in actual_values)
            ]
            if forbidden_matches:
                return (
                    False,
                    matched,
                    f"rejected: excluded {entity_name}={','.join(map(str, forbidden_matches))}",
                )
            if expected_matches:
                matched[entity_name] = expected_matches
                reasons.append(f"{entity_name}={','.join(map(str, expected_matches))}")
            if excluded_values:
                matched[f"excluded_{entity_name}"] = excluded_values
                reasons.append(
                    f"excluded_{entity_name}={','.join(map(str, excluded_values))}"
                )

        availability = normalized_entities.get(
            "availability", normalized_entities.get("in_stock")
        )
        if availability is not None and self._truthy(self._entity_value(availability)):
            actual = product.get(
                "in_stock", product.get("available", product.get("availability"))
            )
            if actual is None:
                return False, matched, "rejected: missing availability metadata"
            if not self._truthy(actual):
                return False, matched, "rejected: unavailable"
            matched["availability"] = True
            reasons.append("availability=in_stock")

        return True, matched, "passed: " + ", ".join(reasons) if reasons else "no_metadata_filters"

    @staticmethod
    def _entity_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            if "value" in value:
                return value["value"]
            if "values" in value:
                return value["values"]
        return value

    @staticmethod
    def _as_values(value: Any) -> list[Any]:
        if isinstance(value, (list, tuple, set)):
            output = []
            for item in value:
                resolved = HybridSearchEngine._entity_value(item)
                if resolved not in (None, ""):
                    output.append(resolved)
            return output
        resolved = HybridSearchEngine._entity_value(value)
        return [] if resolved in (None, "") else [resolved]

    @staticmethod
    def _constraint_values(value: Any) -> tuple[list[Any], list[Any]]:
        items = value if isinstance(value, (list, tuple, set)) else [value]
        included: list[Any] = []
        excluded: list[Any] = []
        for item in items:
            resolved = HybridSearchEngine._entity_value(item)
            if resolved in (None, ""):
                continue
            target = (
                excluded
                if isinstance(item, Mapping) and bool(item.get("negated", False))
                else included
            )
            target.append(resolved)
        return included, excluded

    def _normalized_price_constraint(self, value: Any) -> tuple[float | None, str]:
        if value is None:
            return None, ""
        amount = self._first_number(value)
        if amount is None:
            return None, ""
        currency = "UNKNOWN"
        if isinstance(value, Mapping):
            currency = str(value.get("currency", "UNKNOWN")).upper()
        if currency in {"", "UNKNOWN"}:
            return amount, ""
        rate = self.currency_rates_to_catalog.get(currency)
        if rate is None:
            return None, f"rejected: unsupported price currency {currency}"
        return amount * rate, ""

    @staticmethod
    def _first_number(value: Any) -> float | None:
        if value is None:
            return None
        values = HybridSearchEngine._as_values(value)
        if not values:
            return None
        try:
            return float(values[0])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _metadata_text(value: Any) -> str:
        if isinstance(value, str):
            parsed = json_loads_safe(value, default=value)
            if parsed is not value:
                value = parsed
        if isinstance(value, Mapping):
            value = " ".join(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, (list, tuple, set)):
            value = " ".join(str(item) for item in value)
        return str(value).casefold().replace("_", " ").strip()

    @classmethod
    def _text_matches(cls, expected: Any, actual: str) -> bool:
        expected_text = cls._metadata_text(expected)
        expected_tokens = UNICODE_WORD_PATTERN.findall(expected_text)
        actual_tokens = UNICODE_WORD_PATTERN.findall(cls._metadata_text(actual))
        if not expected_tokens or not actual_tokens:
            return False
        window_size = len(expected_tokens)
        return any(
            all(
                cls._token_equivalent(expected_token, actual_token)
                for expected_token, actual_token in zip(
                    expected_tokens,
                    actual_tokens[start : start + window_size],
                )
            )
            for start in range(len(actual_tokens) - window_size + 1)
        )

    @staticmethod
    def _token_equivalent(expected: str, actual: str) -> bool:
        if expected == actual:
            return True
        if len(expected) <= 3 or len(actual) <= 3:
            return False
        return expected + "s" == actual or actual + "s" == expected

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().casefold() in {
            "1",
            "true",
            "yes",
            "available",
            "in stock",
            "in_stock",
            "có",
            "còn hàng",
        }


# Convenient names for code that models this component as a pipeline.
HybridSearch = HybridSearchEngine
HybridSearchPipeline = HybridSearchEngine


__all__ = [
    "HybridSearch",
    "HybridSearchEngine",
    "HybridSearchPipeline",
    "SEARCH_MODES",
]
