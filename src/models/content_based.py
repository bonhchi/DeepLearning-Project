# Baseline gợi ý dựa trên nội dung.

from __future__ import annotations

from collections import defaultdict

from src.io_utils import parse_int
from src.vector_ops import average_vectors, cosine


# Lấy trung bình embedding của item đã thích để tạo hồ sơ nội dung user.
def build_user_profiles(interactions: list[dict], product_embeddings: dict[str, list[float]]) -> dict[str, list[float]]:
    vectors_by_user: dict[str, list[list[float]]] = defaultdict(list)
    for row in interactions:
        if row.get("split", "train") != "train":
            continue
        if parse_int(row.get("label"), 0) != 1:
            continue
        product_id = row["product_id"]
        if product_id in product_embeddings:
            vectors_by_user[row["user_id"]].append(product_embeddings[product_id])
    return {user_id: average_vectors(vectors) for user_id, vectors in vectors_by_user.items()}


# Xếp hạng sản phẩm chưa thấy theo cosine similarity với hồ sơ user.
def recommend_content(
    user_id: str,
    user_profiles: dict[str, list[float]],
    product_embeddings: dict[str, list[float]],
    seen_product_ids: set[str] | None = None,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    seen = seen_product_ids or set()
    profile = user_profiles.get(user_id)
    if not profile:
        return []
    scored = []
    for product_id, vector in product_embeddings.items():
        if product_id in seen:
            continue
        scored.append((product_id, cosine(profile, vector)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


# Tìm các sản phẩm gần nhất với embedding của sản phẩm mục tiêu.
def similar_items(
    product_id: str,
    product_embeddings: dict[str, list[float]],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    target = product_embeddings.get(product_id)
    if not target:
        return []
    scored = []
    for other_id, vector in product_embeddings.items():
        if other_id == product_id:
            continue
        scored.append((other_id, cosine(target, vector)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
