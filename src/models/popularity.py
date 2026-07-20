# Baseline gợi ý dựa trên độ phổ biến.

from __future__ import annotations

from src.io_utils import parse_float, parse_int


# Ước lượng độ phổ biến sản phẩm từ tương tác positive trong tập train.
def train_popularity(interactions: list[dict]) -> dict[str, float]:
    stats: dict[str, dict[str, float]] = {}
    for row in interactions:
        if row.get("split", "train") != "train":
            continue
        product_id = row["product_id"]
        stats.setdefault(product_id, {"count": 0.0, "rating_sum": 0.0, "weight_sum": 0.0})
        label = parse_int(row.get("label"), 0)
        if label == 1:
            stats[product_id]["count"] += 1.0
            stats[product_id]["rating_sum"] += parse_float(row.get("rating"), 0.0)
            stats[product_id]["weight_sum"] += parse_float(row.get("event_weight"), 1.0)
    scores = {}
    for product_id, values in stats.items():
        count = values["count"]
        avg_rating = values["rating_sum"] / count if count else 0.0
        scores[product_id] = count + 0.15 * avg_rating + 0.05 * values["weight_sum"]
    return scores


# Trả về các sản phẩm phổ biến nhất mà user chưa thấy.
def recommend_popular(
    popularity_scores: dict[str, float],
    product_ids: list[str],
    seen_product_ids: set[str] | None = None,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    seen = seen_product_ids or set()
    scored = []
    for product_id in product_ids:
        if product_id in seen:
            continue
        scored.append((product_id, popularity_scores.get(product_id, 0.0)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
