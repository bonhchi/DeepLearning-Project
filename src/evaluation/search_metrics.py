"""Graded retrieval evaluation for intent-aware search configurations."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

from src.evaluation.metrics import hit_rate_at_k, mrr_at_k, precision_at_k, recall_at_k
from src.io_utils import ensure_parent


SearchFn = Callable[[str, int, dict], list[dict] | list[str]]


def graded_ndcg_at_k(recommended: list[str], relevance: dict[str, int], k: int) -> float:
    dcg = sum(
        (2 ** relevance.get(product_id, 0) - 1) / math.log2(rank + 1)
        for rank, product_id in enumerate(recommended[:k], start=1)
    )
    ideal_values = sorted(relevance.values(), reverse=True)[:k]
    ideal = sum(
        (2**value - 1) / math.log2(rank + 1)
        for rank, value in enumerate(ideal_values, start=1)
    )
    return dcg / ideal if ideal else 0.0


def evaluate_search_configuration(
    queries: Iterable[dict],
    qrels: Iterable[dict],
    search: SearchFn,
    *,
    top_k: int = 5,
    require_reviewed_holdout: bool = True,
) -> dict:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query_rows = [
        row for row in queries if row.get("split") in {"val", "validation", "test"}
    ]
    query_by_id = {str(row.get("query_id")): row for row in query_rows}
    qrels_by_query: dict[str, list[dict]] = defaultdict(list)
    pending_review = 0
    for row in qrels:
        query_id = str(row.get("query_id", ""))
        if query_id not in query_by_id:
            continue
        qrels_by_query[query_id].append(dict(row))

    relevance_by_query: dict[str, dict[str, int]] = {}
    pending_queries = 0
    queries_without_qrels = 0
    for query_id in query_by_id:
        query_qrels = qrels_by_query.get(query_id, [])
        if not query_qrels:
            queries_without_qrels += 1
            if require_reviewed_holdout:
                pending_queries += 1
                continue
            relevance_by_query[query_id] = {}
            continue
        unreviewed = [
            row
            for row in query_qrels
            if str(row.get("reviewed", "false")).casefold()
            not in {"1", "true", "yes"}
        ]
        if require_reviewed_holdout and unreviewed:
            pending_review += len(unreviewed)
            pending_queries += 1
            continue
        relevance: dict[str, int] = {}
        for row in query_qrels:
            value = int(row.get("relevance", 0))
            if value > 0:
                relevance[str(row.get("product_id"))] = value
        relevance_by_query[query_id] = relevance

    totals = defaultdict(float)
    per_intent: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_intent_counts: dict[str, int] = defaultdict(int)
    evaluated = 0
    started = perf_counter()
    for query_id, relevance in relevance_by_query.items():
        query = query_by_id[query_id]
        raw_results = search(str(query.get("query_text", "")), top_k, query)
        product_ids = [
            str(row.get("product_id")) if isinstance(row, dict) else str(row)
            for row in raw_results
        ][:top_k]
        relevant = set(relevance)
        values = {
            "precision": precision_at_k(product_ids, relevant, top_k),
            "recall": recall_at_k(product_ids, relevant, top_k),
            "hit_rate": hit_rate_at_k(product_ids, relevant, top_k),
            "mrr": mrr_at_k(product_ids, relevant, top_k),
            "ndcg": graded_ndcg_at_k(product_ids, relevance, top_k),
        }
        intent = str(query.get("intent", "unknown"))
        for name, value in values.items():
            totals[name] += value
            per_intent[intent][name] += value
        per_intent_counts[intent] += 1
        evaluated += 1
    elapsed_ms = (perf_counter() - started) * 1000.0
    aggregate = {
        f"{name}@{top_k}": round(value / max(evaluated, 1), 6)
        for name, value in totals.items()
    }
    aggregate.update(
        {
            "queries_evaluated": evaluated,
            "pending_manual_review": pending_review,
            "queries_pending_manual_review": pending_queries,
            "queries_without_qrels": queries_without_qrels,
            "latency_ms_per_query": round(elapsed_ms / max(evaluated, 1), 3),
        }
    )
    intent_metrics = {
        intent: {
            **{
                f"{name}@{top_k}": round(value / per_intent_counts[intent], 6)
                for name, value in values.items()
            },
            "queries_evaluated": per_intent_counts[intent],
        }
        for intent, values in sorted(per_intent.items())
    }
    return {"aggregate": aggregate, "per_intent": intent_metrics}


def compare_search_configurations(
    queries: Iterable[dict],
    qrels: Iterable[dict],
    searchers: dict[str, SearchFn],
    *,
    top_k: int = 5,
    require_reviewed_holdout: bool = True,
) -> dict:
    query_rows = list(queries)
    qrel_rows = list(qrels)
    return {
        name: evaluate_search_configuration(
            query_rows,
            qrel_rows,
            searcher,
            top_k=top_k,
            require_reviewed_holdout=require_reviewed_holdout,
        )
        for name, searcher in searchers.items()
    }


def save_search_metrics(json_path: str | Path, csv_path: str | Path, metrics: dict) -> None:
    ensure_parent(json_path)
    ensure_parent(csv_path)
    Path(json_path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    metric_names = sorted(
        {
            key
            for result in metrics.values()
            for key in result.get("aggregate", {})
        }
        | {
            key
            for result in metrics.values()
            for intent_metrics in result.get("per_intent", {}).values()
            for key in intent_metrics
        }
    )
    with Path(csv_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["configuration", "scope", "intent", *metric_names],
        )
        writer.writeheader()
        for name, result in metrics.items():
            writer.writerow(
                {
                    "configuration": name,
                    "scope": "aggregate",
                    "intent": "all",
                    **result.get("aggregate", {}),
                }
            )
            for intent, values in sorted(result.get("per_intent", {}).items()):
                writer.writerow(
                    {
                        "configuration": name,
                        "scope": "per_intent",
                        "intent": intent,
                        **values,
                    }
                )
