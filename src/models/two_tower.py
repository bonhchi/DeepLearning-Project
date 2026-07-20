# Model retrieval two-tower nhẹ, viết bằng Python thuần.

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from src.io_utils import ensure_parent, parse_int
from src.vector_ops import dot, normalize, project_vector, sigmoid, stable_hash_int, stable_random_vector


# Train embedding user và item bằng event positive và negative được lấy mẫu.
class TwoTowerModel:

    # Khởi tạo model two-tower rỗng.
    def __init__(self, dim: int = 48, regularization: float = 0.0005) -> None:
        self.dim = dim
        self.regularization = regularization
        self.user_embeddings: dict[str, list[float]] = {}
        self.item_embeddings: dict[str, list[float]] = {}
        self.item_bias: dict[str, float] = {}

    # Khởi tạo vector item tower từ đặc trưng đa phương thức đã hợp nhất.
    def _init_item(self, product_id: str, product_embeddings: dict[str, list[float]]) -> None:
        if product_id in self.item_embeddings:
            return
        base = product_embeddings.get(product_id)
        self.item_embeddings[product_id] = project_vector(base, self.dim) if base else stable_random_vector(product_id, self.dim, 0.2)
        self.item_bias[product_id] = 0.0

    # Khởi tạo vector user tower từ item đã thích hoặc seed ngẫu nhiên ổn định.
    def _init_user(self, user_id: str, positive_items: list[str]) -> None:
        if user_id in self.user_embeddings:
            return
        vectors = [self.item_embeddings[item_id] for item_id in positive_items if item_id in self.item_embeddings]
        if vectors:
            output = [0.0] * self.dim
            for vector in vectors:
                for index, value in enumerate(vector):
                    output[index] += value
            self.user_embeddings[user_id] = normalize([value / len(vectors) for value in output])
        else:
            self.user_embeddings[user_id] = stable_random_vector(user_id, self.dim, 0.2)

    # Cập nhật SGD một bước cho một cặp user-item theo logistic loss.
    def _update_pair(self, user_id: str, product_id: str, label: int, learning_rate: float) -> None:
        user_vector = self.user_embeddings[user_id]
        item_vector = self.item_embeddings[product_id]
        bias = self.item_bias.get(product_id, 0.0)
        prediction = sigmoid(dot(user_vector, item_vector) + bias)
        error = label - prediction
        for index in range(self.dim):
            old_user_value = user_vector[index]
            old_item_value = item_vector[index]
            user_vector[index] += learning_rate * (error * old_item_value - self.regularization * old_user_value)
            item_vector[index] += learning_rate * (error * old_user_value - self.regularization * old_item_value)
        self.item_bias[product_id] = bias + learning_rate * (error - self.regularization * bias)

    # Train user tower và item tower từ các tương tác positive trong tập train.
    def fit(
        self,
        interactions: list[dict],
        product_embeddings: dict[str, list[float]],
        epochs: int = 3,
        negative_samples: int = 2,
        learning_rate: float = 0.04,
    ) -> "TwoTowerModel":
        positives = [
            row
            for row in interactions
            if row.get("split", "train") == "train" and parse_int(row.get("label"), 0) == 1
        ]
        positive_by_user: dict[str, set[str]] = defaultdict(set)
        for row in positives:
            positive_by_user[row["user_id"]].add(row["product_id"])
            self._init_item(row["product_id"], product_embeddings)
        for product_id in product_embeddings:
            self._init_item(product_id, product_embeddings)
        for user_id, item_ids in positive_by_user.items():
            self._init_user(user_id, list(item_ids))

        all_items = list(self.item_embeddings)
        rng = random.Random(42)
        for epoch in range(max(epochs, 1)):
            rng.shuffle(positives)
            epoch_learning_rate = learning_rate / (1.0 + 0.2 * epoch)
            for row in positives:
                user_id = row["user_id"]
                positive_item = row["product_id"]
                self._update_pair(user_id, positive_item, 1, epoch_learning_rate)
                for _ in range(max(negative_samples, 0)):
                    negative_item = self._sample_negative(all_items, positive_by_user[user_id], rng)
                    if negative_item:
                        self._update_pair(user_id, negative_item, 0, epoch_learning_rate)
        return self

    # Lấy mẫu một item mà user chưa có tương tác positive.
    def _sample_negative(self, all_items: list[str], positive_items: set[str], rng: random.Random) -> str:
        if not all_items:
            return ""
        for _ in range(20):
            candidate = all_items[rng.randrange(len(all_items))]
            if candidate not in positive_items:
                return candidate
        return all_items[stable_hash_int(str(len(positive_items)), len(all_items))]

    # Tính điểm cho một cặp user-product bằng dot product cộng item bias.
    def score(self, user_id: str, product_id: str) -> float:
        user_vector = self.user_embeddings.get(user_id)
        item_vector = self.item_embeddings.get(product_id)
        if not user_vector or not item_vector:
            return 0.0
        return dot(user_vector, item_vector) + self.item_bias.get(product_id, 0.0)

    # Xếp hạng sản phẩm chưa thấy cho user bằng model two-tower đã train.
    def recommend(
        self,
        user_id: str,
        seen_product_ids: set[str] | None = None,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        seen = seen_product_ids or set()
        scored = []
        for product_id in self.item_embeddings:
            if product_id in seen:
                continue
            scored.append((product_id, self.score(user_id, product_id)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    # Serialize model thành dictionary tương thích JSON.
    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "regularization": self.regularization,
            "user_embeddings": self.user_embeddings,
            "item_embeddings": self.item_embeddings,
            "item_bias": self.item_bias,
        }

    # Khôi phục model từ dictionary tương thích JSON.
    @classmethod
    def from_dict(cls, data: dict) -> "TwoTowerModel":
        model = cls(dim=int(data.get("dim", 48)), regularization=float(data.get("regularization", 0.0005)))
        model.user_embeddings = {str(key): list(value) for key, value in data.get("user_embeddings", {}).items()}
        model.item_embeddings = {str(key): list(value) for key, value in data.get("item_embeddings", {}).items()}
        model.item_bias = {str(key): float(value) for key, value in data.get("item_bias", {}).items()}
        return model

    # Lưu model thành artifact JSON.
    def save(self, path: str | Path) -> None:
        ensure_parent(path)
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    # Load artifact model JSON đã lưu.
    @classmethod
    def load(cls, path: str | Path) -> "TwoTowerModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
