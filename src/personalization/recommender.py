# Bộ gợi ý cấp cao dùng cho CLI và demo Streamlit.

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.config import ProjectConfig
from src.io_utils import json_loads_safe, parse_float, parse_int, read_csv_rows, read_jsonl
from src.models.content_based import build_user_profiles, recommend_content, similar_items
from src.models.popularity import recommend_popular, train_popularity
from src.models.two_tower import TwoTowerModel


# Kết hợp model đã train, baseline, catalog và phần giải thích.
class PersonalizedRecommender:

    # Khởi tạo các bảng tra cứu cần cho việc phục vụ gợi ý cá nhân hóa.
    def __init__(
        self,
        users: list[dict],
        products: list[dict],
        interactions: list[dict],
        product_embeddings: dict[str, list[float]],
        model: TwoTowerModel | None = None,
    ) -> None:
        self.users = {row["user_id"]: row for row in users}
        self.products = {row["product_id"]: row for row in products}
        self.product_ids = list(self.products)
        self.interactions = interactions
        self.product_embeddings = product_embeddings
        self.model = model
        self.popularity_scores = train_popularity(interactions)
        self.user_profiles = build_user_profiles(interactions, product_embeddings)
        self.seen_by_user = self._build_seen_items(interactions)

    # Load dữ liệu đã xử lý và artifact model từ thư mục dự án.
    @classmethod
    def from_project(cls, project_root: str | Path) -> "PersonalizedRecommender":
        config = ProjectConfig(project_root=Path(project_root).resolve())
        users = read_csv_rows(config.users_path)
        products = read_csv_rows(config.products_path)
        interactions = read_csv_rows(config.interactions_path)
        embeddings = {
            row["product_id"]: row["fused_embedding"]
            for row in read_jsonl(config.product_embeddings_jsonl_path)
        }
        model = TwoTowerModel.load(config.two_tower_model_path) if config.two_tower_model_path.exists() else None
        return cls(users, products, interactions, embeddings, model)

    # Tạo map user-to-products để lọc các sản phẩm user đã xem.
    def _build_seen_items(self, interactions: list[dict]) -> dict[str, set[str]]:
        seen: dict[str, set[str]] = defaultdict(set)
        for row in interactions:
            if row.get("split", "train") == "train":
                seen[row["user_id"]].add(row["product_id"])
        return seen

    # Sinh gợi ý theo content và fallback sang popularity khi cần.
    def _fallback_scores(self, user_id: str, top_k: int) -> list[tuple[str, float]]:
        seen = self.seen_by_user.get(user_id, set())
        content_rows = recommend_content(user_id, self.user_profiles, self.product_embeddings, seen, top_k)
        if content_rows:
            return content_rows
        return recommend_popular(self.popularity_scores, self.product_ids, seen, top_k)

    # Gắn thông tin catalog và lý do giải thích cho một gợi ý.
    def _enrich(self, product_id: str, score: float, user_id: str) -> dict:
        product = dict(self.products.get(product_id, {}))
        product["product_id"] = product_id
        product["score"] = float(score)
        product["explanation"] = self.explain_recommendation(user_id, product_id)
        return product

    # Trả về Top-K sản phẩm cá nhân hóa kèm điểm và giải thích.
    def recommend_for_user(self, user_id: str, top_k: int = 10) -> list[dict]:
        seen = self.seen_by_user.get(user_id, set())
        if self.model and user_id in self.model.user_embeddings:
            rows = self.model.recommend(user_id, seen, top_k)
        else:
            rows = self._fallback_scores(user_id, top_k)
        if not rows:
            rows = recommend_popular(self.popularity_scores, self.product_ids, seen, top_k)
        return [self._enrich(product_id, score, user_id) for product_id, score in rows]

    # Tạo giải thích ngắn từ hồ sơ user và metadata sản phẩm.
    def explain_recommendation(self, user_id: str, product_id: str) -> str:
        user = self.users.get(user_id, {})
        product = self.products.get(product_id, {})
        preferred_categories = json_loads_safe(user.get("preferred_categories"), default=[])
        category = product.get("category", "fashion")
        price_range = product.get("preferred_price_range", "unknown")
        rating = parse_float(product.get("average_rating"), 0.0)
        if category in preferred_categories:
            return f"Matches your interest in {category} and has avg rating {rating:.1f}."
        if user.get("preferred_price_range") == price_range and price_range != "unknown":
            return f"Fits your usual {price_range} price range with avg rating {rating:.1f}."
        return f"Recommended from similar shopping behavior; avg rating {rating:.1f}."

    # Trả về các sản phẩm tương tự nhất với item được chọn.
    def similar_products(self, product_id: str, top_k: int = 10) -> list[dict]:
        rows = similar_items(product_id, self.product_embeddings, top_k)
        return [self._enrich(other_id, score, "") for other_id, score in rows]

    # Gợi ý bundle nhỏ bằng cách kết hợp nhiều category được đề xuất.
    def bundle_suggestion(self, user_id: str, top_k: int = 5) -> list[dict]:
        recommendations = self.recommend_for_user(user_id, top_k=max(top_k * 2, 4))
        bundle = []
        used_categories = set()
        for item in recommendations:
            category = item.get("category", "fashion")
            if category in used_categories:
                continue
            used_categories.add(category)
            bundle.append(item)
            if len(bundle) >= top_k:
                break
        return bundle
