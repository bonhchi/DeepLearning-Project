"""Build and validate graded relevance judgments for product retrieval."""

from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.io_utils import ensure_parent


LOGGER = logging.getLogger(__name__)
QREL_COLUMNS = ["query_id", "product_id", "relevance", "source", "reviewed"]
REVIEW_COLUMNS = [
    *QREL_COLUMNS,
    "query_text",
    "intent",
    "split",
    "product_title",
    "auto_reason",
    "user_id",
    "profile_context",
]
QUERY_STOPWORDS = {
    "a", "an", "and", "based", "best", "cho", "compare", "dựa", "find",
    "for", "gợi", "hãy", "is", "khác", "me", "other", "phẩm", "product",
    "recommend", "sản", "show", "similar", "so", "sở", "thích", "tìm",
    "to", "tôi", "trên", "với",
}


def _tokens(value: object) -> set[str]:
    return set(re.findall(r"[\w]+", str(value).casefold(), flags=re.UNICODE))


def build_qrels(
    queries: Iterable[dict],
    products: Iterable[dict],
    *,
    max_candidates_per_query: int = 20,
    candidate_pools: Mapping[str, Sequence[str]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Create automatic candidates and a manual-review queue for val/test.

    Automatic labels are suitable for training/bootstrap only. Validation and test
    candidates are emitted with ``reviewed=false`` and must be confirmed before a
    benchmark can claim retrieval quality.
    """

    if max_candidates_per_query <= 0:
        raise ValueError("max_candidates_per_query must be positive")
    product_rows = [row for row in products if row.get("product_id")]
    product_by_id = {str(row["product_id"]): row for row in product_rows}
    product_tokens: dict[str, set[str]] = {}
    token_postings: dict[str, set[str]] = defaultdict(set)
    for product_id, product in product_by_id.items():
        text = " ".join(
            str(product.get(field, ""))
            for field in ("title", "category", "store", "description", "features")
        )
        tokens = _tokens(text)
        product_tokens[product_id] = tokens
        for token in tokens:
            token_postings[token].add(product_id)
    qrels: list[dict] = []
    review_queue: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for query in queries:
        query_id = str(query.get("query_id", "")).strip()
        query_text = str(query.get("query_text", "")).strip()
        if not query_id or not query_text:
            continue
        category_tokens = _tokens(query.get("category", ""))
        query_tokens = _tokens(query_text) - category_tokens - QUERY_STOPWORDS
        category = str(query.get("category", "")).strip()
        source = str(query.get("source", ""))
        source_product_id = source.rsplit(":", 1)[-1] if ":" in source else ""
        intent = str(query.get("intent", ""))
        pooled_ids = {
            str(product_id)
            for product_id in (candidate_pools or {}).get(query_id, [])
            if str(product_id) in product_by_id
        }
        candidate_ids = set(pooled_ids)
        if source_product_id in product_by_id:
            candidate_ids.add(source_product_id)
        for token in query_tokens:
            candidate_ids.update(token_postings.get(token, ()))

        scored: list[tuple[int, int, float, dict, str]] = []
        for product_id in candidate_ids:
            product = product_by_id[product_id]
            overlap = len(query_tokens & product_tokens[product_id]) / max(len(query_tokens), 1)
            category_match = bool(category and str(product.get("category", "")) == category)
            if intent == "similar_product_search" and product_id == source_product_id:
                # The anchor is context, not a valid "similar product" answer.
                continue
            if product_id == source_product_id and intent == "availability_check":
                raw_availability = product.get(
                    "in_stock", product.get("available", product.get("availability", ""))
                )
                normalized_availability = str(raw_availability).strip().casefold()
                if normalized_availability in {
                    "1", "true", "yes", "available", "in stock", "còn hàng", "con hang"
                }:
                    relevance = 2
                    reason = (
                        "available_source_product_demo_inventory_proxy"
                        if product.get("availability_source")
                        == "business_context_inventory_proxy"
                        else "available_source_product"
                    )
                elif normalized_availability in {
                    "0", "false", "no", "unavailable", "out of stock", "hết hàng", "het hang"
                }:
                    relevance = 0
                    reason = (
                        "unavailable_source_product_demo_inventory_proxy"
                        if product.get("availability_source")
                        == "business_context_inventory_proxy"
                        else "unavailable_source_product"
                    )
                else:
                    relevance = 0
                    reason = "unknown_availability_source_product"
            elif product_id == source_product_id:
                relevance = 2
                reason = "source_product"
            elif category_match and overlap > 0:
                relevance = 1
                reason = "metadata_overlap"
            elif product_id in pooled_ids:
                relevance = 0
                reason = "retrieval_pool"
            else:
                continue
            selection_priority = (
                0 if product_id == source_product_id else 1 if product_id in pooled_ids else 2
            )
            scored.append((selection_priority, relevance, overlap, product, reason))
        scored.sort(
            key=lambda row: (
                row[0],
                -row[1],
                -row[2],
                str(row[3].get("product_id", "")),
            )
        )
        for _, relevance, _, product, reason in scored[:max_candidates_per_query]:
            product_id = str(product["product_id"])
            key = (query_id, product_id)
            if key in seen:
                continue
            seen.add(key)
            needs_review = (
                query.get("split") in {"val", "validation", "test"}
                or reason == "unknown_availability_source_product"
            )
            row = {
                "query_id": query_id,
                "product_id": product_id,
                "relevance": relevance,
                "source": reason,
                "reviewed": "false" if needs_review else "true",
            }
            qrels.append(row)
            if needs_review:
                review_queue.append(
                    row
                    | {
                        "query_text": query_text,
                        "intent": query.get("intent", ""),
                        "split": query.get("split", ""),
                        "product_title": product.get("title", ""),
                        "auto_reason": reason,
                    }
                )
    LOGGER.info("Built %s qrels; %s require review", len(qrels), len(review_queue))
    return qrels, review_queue


def merge_review_queue(
    fresh_rows: Iterable[dict],
    existing_rows: Iterable[dict],
) -> tuple[list[dict], int]:
    """Preserve completed judgments for pairs still present in a rebuilt queue."""

    fresh = [dict(row) for row in fresh_rows]
    existing_by_pair = {
        (str(row.get("query_id", "")), str(row.get("product_id", ""))): dict(row)
        for row in existing_rows
        if str(row.get("reviewed", "false")).casefold() in {"1", "true", "yes"}
    }
    preserved = 0
    for row in fresh:
        key = (str(row.get("query_id", "")), str(row.get("product_id", "")))
        previous = existing_by_pair.get(key)
        if previous is None:
            continue
        row["relevance"] = previous.get("relevance", row.get("relevance", 0))
        row["source"] = "manual_review"
        row["reviewed"] = "true"
        preserved += 1
    validate_qrels(fresh)
    return fresh, preserved


def validate_qrels(
    rows: Iterable[dict],
    *,
    query_ids: set[str] | None = None,
    product_ids: set[str] | None = None,
) -> None:
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=2):
        query_id = str(row.get("query_id", "")).strip()
        product_id = str(row.get("product_id", "")).strip()
        if not query_id or not product_id:
            raise ValueError(f"Missing qrel identifier at CSV row {index}")
        key = (query_id, product_id)
        if key in seen:
            raise ValueError(f"Duplicate qrel pair: {key}")
        seen.add(key)
        try:
            relevance = int(row.get("relevance", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid relevance at CSV row {index}") from exc
        if relevance not in {0, 1, 2}:
            raise ValueError(f"Relevance must be 0, 1 or 2 at CSV row {index}")
        if query_ids is not None and query_id not in query_ids:
            raise ValueError(f"Unknown query_id: {query_id}")
        if product_ids is not None and product_id not in product_ids:
            raise ValueError(f"Unknown product_id: {product_id}")


def write_qrels(path: str | Path, rows: Iterable[dict], *, review: bool = False) -> int:
    output = list(rows)
    validate_qrels(output)
    ensure_parent(path)
    columns = REVIEW_COLUMNS if review else QREL_COLUMNS
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    return len(output)
