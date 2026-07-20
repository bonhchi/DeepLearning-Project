# Các metric xếp hạng dùng để đánh giá gợi ý Top-K.

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable

from src.io_utils import parse_int


RecommendationFn = Callable[[str, set[str], int], list[tuple[str, float]]]


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


# Chạy nhiều recommender trên cùng tập user test giữ lại.
def evaluate_recommenders(
    interactions: list[dict],
    recommenders: dict[str, RecommendationFn],
    top_k: int = 10,
) -> dict[str, dict[str, float]]:
    seen_by_user, relevant_by_user = build_eval_sets(interactions)
    results = {}
    for name, recommender in recommenders.items():
        recommendations_by_user = {}
        for user_id in relevant_by_user:
            rows = recommender(user_id, seen_by_user.get(user_id, set()), top_k)
            recommendations_by_user[user_id] = [product_id for product_id, _ in rows]
        results[name] = evaluate_recommendations(recommendations_by_user, relevant_by_user, top_k)
    return results
