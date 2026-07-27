# Các metric xếp hạng dùng để đánh giá gợi ý Top-K.

from __future__ import annotations

import math
from collections import defaultdict
from time import perf_counter
from typing import Callable

from src.io_utils import parse_int


RecommendationFn = Callable[[str, set[str], int], list[tuple[str, float]]]


def artifact_catalog_health(
    artifact_product_ids: set[str],
    catalog_product_ids: set[str],
) -> dict[str, float]:
    """Kiểm tra artifact model/embedding còn đồng bộ với catalog hiện tại hay không."""
    return {
        "artifact_catalog_coverage": round(
            len(artifact_product_ids & catalog_product_ids) / max(len(catalog_product_ids), 1),
            6,
        ),
        "artifact_out_of_catalog_rate": round(
            len(artifact_product_ids - catalog_product_ids) / max(len(artifact_product_ids), 1),
            6,
        ),
    }


# Tính Precision@K cho một user.
def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / k


# Tính Recall@K cho một user.
def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / len(relevant)


# Tính HitRate@K cho một user.
def hit_rate_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


# Tính NDCG@K nhị phân cho một user.
def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for index, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


# Tính reciprocal rank của item liên quan đầu tiên trong Top-K.
def mrr_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    for index, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


# Tạo tập item train đã thấy và tập positive test giữ lại cho mỗi user.
def build_eval_sets(interactions: list[dict]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    seen_by_user: dict[str, set[str]] = defaultdict(set)
    relevant_by_user: dict[str, set[str]] = defaultdict(set)
    for row in interactions:
        user_id = row["user_id"]
        product_id = row["product_id"]
        if row.get("split", "train") == "train":
            seen_by_user[user_id].add(product_id)
        if row.get("split") == "test" and parse_int(row.get("label"), 0) == 1:
            relevant_by_user[user_id].add(product_id)
    return seen_by_user, relevant_by_user


# Tổng hợp các metric xếp hạng trên toàn bộ user có thể đánh giá.
def evaluate_recommendations(
    recommendations_by_user: dict[str, list[str]],
    relevant_by_user: dict[str, set[str]],
    top_k: int = 10,
) -> dict[str, float]:
    totals = {
        "precision": 0.0,
        "recall": 0.0,
        "hit_rate": 0.0,
        "ndcg": 0.0,
        "mrr": 0.0,
    }
    users = [user_id for user_id in relevant_by_user if relevant_by_user[user_id]]
    if not users:
        return {f"{name}@{top_k}": 0.0 for name in totals} | {"users_evaluated": 0}
    for user_id in users:
        recommended = recommendations_by_user.get(user_id, [])
        relevant = relevant_by_user[user_id]
        totals["precision"] += precision_at_k(recommended, relevant, top_k)
        totals["recall"] += recall_at_k(recommended, relevant, top_k)
        totals["hit_rate"] += hit_rate_at_k(recommended, relevant, top_k)
        totals["ndcg"] += ndcg_at_k(recommended, relevant, top_k)
        totals["mrr"] += mrr_at_k(recommended, relevant, top_k)
    return {f"{name}@{top_k}": round(value / len(users), 6) for name, value in totals.items()} | {
        "users_evaluated": len(users)
    }


def evaluate_serving_health(
    recommendations_by_user: dict[str, list[str]],
    seen_by_user: dict[str, set[str]],
    relevant_by_user: dict[str, set[str]],
    catalog_product_ids: set[str],
    top_k: int,
    elapsed_ms: float,
) -> dict[str, float]:
    """Đo coverage, trùng lặp, rò rỉ train và độ trễ của output recommender."""
    users = [user_id for user_id in relevant_by_user if relevant_by_user[user_id]]
    lists = [recommendations_by_user.get(user_id, [])[:top_k] for user_id in users]
    total_recommended = sum(len(rows) for rows in lists)
    unique_recommended = {product_id for rows in lists for product_id in rows}
    duplicate_count = sum(len(rows) - len(set(rows)) for rows in lists)
    seen_leakage = sum(
        product_id in seen_by_user.get(user_id, set())
        for user_id, rows in zip(users, lists)
        for product_id in rows
    )
    out_of_catalog = sum(
        product_id not in catalog_product_ids for rows in lists for product_id in rows
    )
    relevant_items = {
        product_id for user_id in users for product_id in relevant_by_user.get(user_id, set())
    }
    return {
        f"catalog_coverage@{top_k}": round(
            len(unique_recommended & catalog_product_ids) / max(len(catalog_product_ids), 1), 6
        ),
        f"avg_list_size@{top_k}": round(total_recommended / max(len(users), 1), 6),
        f"duplicate_rate@{top_k}": round(duplicate_count / max(total_recommended, 1), 6),
        f"seen_leakage_rate@{top_k}": round(seen_leakage / max(total_recommended, 1), 6),
        f"out_of_catalog_rate@{top_k}": round(out_of_catalog / max(total_recommended, 1), 6),
        "relevant_in_catalog_rate": round(
            len(relevant_items & catalog_product_ids) / max(len(relevant_items), 1), 6
        ),
        "latency_ms_per_user": round(elapsed_ms / max(len(users), 1), 3),
    }
# Chạy nhiều recommender trên cùng tập user test giữ lại.
def evaluate_recommenders(
    interactions: list[dict],
    recommenders: dict[str, RecommendationFn],
    top_k: int = 10,
    catalog_product_ids: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    seen_by_user, relevant_by_user = build_eval_sets(interactions)
    catalog = catalog_product_ids or {
        row["product_id"] for row in interactions if row.get("product_id")
    }
    results = {}
    for name, recommender in recommenders.items():
        started_at = perf_counter()
        recommendations_by_user = {}
        for user_id in relevant_by_user:
            rows = recommender(user_id, seen_by_user.get(user_id, set()), top_k)
            recommendations_by_user[user_id] = [product_id for product_id, _ in rows]
        elapsed_ms = (perf_counter() - started_at) * 1000
        results[name] = evaluate_recommendations(
            recommendations_by_user, relevant_by_user, top_k
        ) | evaluate_serving_health(
            recommendations_by_user,
            seen_by_user,
            relevant_by_user,
            catalog,
            top_k,
            elapsed_ms,
        )
    return results
