# Bộ gợi ý cấp cao dùng cho CLI và demo Streamlit.

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from src.config import ProjectConfig
from src.feature_extraction.embeddings import product_text, tokenize
from src.io_utils import (
    json_loads_safe,
    parse_float,
    parse_int,
    read_csv_rows,
    read_json,
    read_jsonl,
)
from src.models.content_based import build_user_profiles, recommend_content
from src.models.popularity import recommend_popular, train_popularity
from src.models.two_tower import TwoTowerModel
from src.vector_ops import cosine, sigmoid


NEED_CATEGORY_ALIASES = {
    "Appliances": {
        "appliance", "appliances", "refrigerator", "fridge", "freezer", "dishwasher",
        "washing machine", "tu lanh", "may giat", "may rua bat", "do dien gia dung",
    },
    "Automotive": {
        "automotive", "car", "cars", "vehicle", "motor", "tire", "oil", "oto", "xe", "xe hoi",
    },
    "Electronics": {
        "electronics", "electronic", "phone", "computer", "audio", "camera", "dien tu", "may tinh",
        "dien thoai", "tai nghe", "loa",
    },
    "Health_and_Household": {
        "health", "household", "vitamin", "supplement", "cleaning", "suc khoe", "gia dung", "ve sinh",
    },
    "Beauty_and_Personal_Care": {
        "beauty", "personal care", "skin", "hair", "makeup", "shampoo", "lam dep", "cham soc ca nhan",
        "cham soc da", "my pham",
    },
    "Fashion": {"fashion", "thoi trang"},
    "dresses": {"dress", "dresses", "vay"},
    "shoes": {"shoe", "shoes", "sneaker", "boot", "giay"},
    "tops": {"shirt", "shirts", "t shirt", "t-shirt", "blouse", "top", "ao thun", "ao phong"},
    "outerwear": {"jacket", "coat", "outerwear", "ao khoac"},
    "bottoms": {"pants", "jeans", "trousers", "bottoms", "quan"},
    "socks": {"sock", "socks", "vo"},
    "bags": {"bag", "bags", "purse", "tui"},
    "watches": {"watch", "watches", "dong ho"},
    "eyewear": {"glasses", "sunglasses", "eyewear", "kinh"},
    "jewelry": {"jewelry", "necklace", "ring", "earring", "trang suc"},
}

# Một nhu cầu con của Fashion có thể được lưu bằng nhãn con (dữ liệu local cũ)
# hoặc nhãn Amazon_Fashion chung (metadata/review category chính thức).
FASHION_CATALOG_CATEGORIES = (
    "tops", "dresses", "shoes", "outerwear", "bottoms", "socks", "bags", "watches",
    "eyewear", "jewelry", "fashion_accessories", "Amazon_Fashion", "Fashion",
)
NEED_CATEGORY_COMPATIBILITY = {
    "Fashion": FASHION_CATALOG_CATEGORIES,
    "dresses": ("dresses", "Amazon_Fashion", "Fashion"),
    "shoes": ("shoes", "Amazon_Fashion", "Fashion"),
    "tops": ("tops", "Amazon_Fashion", "Fashion"),
    "outerwear": ("outerwear", "Amazon_Fashion", "Fashion"),
    "bottoms": ("bottoms", "Amazon_Fashion", "Fashion"),
    "socks": ("socks", "Amazon_Fashion", "Fashion"),
    "bags": ("bags", "Amazon_Fashion", "Fashion"),
    "watches": ("watches", "Amazon_Fashion", "Fashion"),
    "eyewear": ("eyewear", "Amazon_Fashion", "Fashion"),
    "jewelry": ("jewelry", "Amazon_Fashion", "Fashion"),
}

CATEGORY_LABELS = {
    "Appliances": "điện gia dụng",
    "Automotive": "ô tô và xe",
    "Electronics": "điện tử",
    "Health_and_Household": "sức khỏe và gia dụng",
    "Beauty_and_Personal_Care": "làm đẹp và chăm sóc cá nhân",
    "dresses": "váy",
    "shoes": "giày",
    "tops": "áo",
    "outerwear": "áo khoác",
    "bottoms": "quần",
    "socks": "vớ",
    "bags": "túi",
    "watches": "đồng hồ",
    "eyewear": "kính",
    "jewelry": "trang sức",
}

NEED_QUERY_STOPWORDS = {
    "toi", "can", "muon", "mot", "san", "pham", "duoc", "tot", "good", "need", "want", "product",
    "looking", "useful", "current",
}

NEED_TOKEN_EXPANSIONS = {
    "thun": {"shirt", "tshirt", "tee"},
    "vay": {"dress"},
    "giay": {"shoe", "sneaker"},
    "tui": {"bag", "purse"},
    "lanh": {"refrigerator", "fridge"},
}

NEED_PHRASE_EXPANSIONS = {
    "tai nghe": {"headphone", "headphones", "earbud", "earbuds", "earphone"},
    "dien thoai": {"phone", "smartphone"},
    "may tinh": {"computer", "laptop", "desktop"},
    "cham soc da": {"skincare", "skin", "cream", "serum"},
    "ao thun": {"shirt", "tshirt", "tee"},
}


def _ascii_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return without_accents.replace("đ", "d")


def _unit_cosine(left: list[float], right: list[float]) -> float:
    return max(0.0, min(1.0, (cosine(left, right) + 1.0) / 2.0))


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
        modality_embeddings: dict[str, dict[str, list[float]]] | None = None,
        vocabulary: list[str] | None = None,
    ) -> None:
        self.users = {row["user_id"]: row for row in users}
        self.products = {row["product_id"]: row for row in products}
        self.product_ids = list(self.products)
        self.interactions = interactions
        self.product_embeddings = product_embeddings
        self.model = model
        self.modality_embeddings = modality_embeddings or {}
        self.vocabulary = vocabulary or []
        self.popularity_scores = train_popularity(interactions)
        self.max_popularity = max(self.popularity_scores.values(), default=1.0)
        # Serving mode không nạp embedding, vì vậy không cần quét interactions lần nữa.
        self.user_profiles = (
            build_user_profiles(interactions, product_embeddings) if product_embeddings else {}
        )
        self.seen_by_user = self._build_seen_items(interactions)
        self.active_user_ids = sorted(
            self.users,
            key=lambda user_id: len(self.seen_by_user.get(user_id, set())),
            reverse=True,
        )[:500]
        self.products_by_category = self._index_products_by_category(products)
        self.product_search_preview = {
            product_id: _ascii_text(
                " ".join(
                    [
                        str(product.get("title", "")),
                        str(product.get("features", "")),
                    ]
                )
            )
            for product_id, product in self.products.items()
        }

    # Load dữ liệu đã xử lý và artifact model từ thư mục dự án.
    @classmethod
    def from_project(
        cls,
        project_root: str | Path,
        load_embeddings: bool = False,
    ) -> "PersonalizedRecommender":
        config = ProjectConfig(project_root=Path(project_root).resolve())
        users = read_csv_rows(config.users_path)
        products = read_csv_rows(config.products_path)
        interactions = read_csv_rows(config.interactions_path)
        embeddings: dict[str, list[float]] = {}
        modality_embeddings: dict[str, dict[str, list[float]]] = {}
        if load_embeddings:
            embedding_rows = list(read_jsonl(config.product_embeddings_jsonl_path))
            embeddings = {row["product_id"]: row["fused_embedding"] for row in embedding_rows}
            modality_embeddings = {
                row["product_id"]: {
                    "text": row.get("text_embedding", []),
                    "image": row.get("image_embedding", []),
                    "metadata": row.get("metadata_embedding", []),
                    "fused": row.get("fused_embedding", []),
                }
                for row in embedding_rows
            }
        vocabulary = read_json(config.text_vocabulary_path, default=[])
        model = TwoTowerModel.load(config.two_tower_model_path) if config.two_tower_model_path.exists() else None
        return cls(users, products, interactions, embeddings, model, modality_embeddings, vocabulary)

    # Lập chỉ mục category để giới hạn số item phải chấm điểm cho một nhu cầu rõ ràng.
    def _index_products_by_category(self, products: list[dict]) -> dict[str, list[str]]:
        output: dict[str, list[str]] = defaultdict(list)
        for product in products:
            output[str(product.get("category", "unknown"))].append(product["product_id"])
        return dict(output)

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

    # Nhận diện domain từ câu nhu cầu tiếng Việt hoặc tiếng Anh.
    def infer_need_categories(self, need: str) -> list[str]:
        normalized = _ascii_text(need)
        words = set(normalized.split())
        return [
            category
            for category, aliases in NEED_CATEGORY_ALIASES.items()
            if any(
                alias in words if " " not in alias and len(alias) <= 3 else alias in normalized
                for alias in aliases
            )
        ]

    def resolve_need_categories(self, need: str) -> tuple[list[str], list[str]]:
        """Trả về category catalog phù hợp và intent chưa có dữ liệu."""
        inferred = self.infer_need_categories(need)
        resolved: list[str] = []
        missing: list[str] = []
        for intent in inferred:
            compatible = NEED_CATEGORY_COMPATIBILITY.get(intent, (intent,))
            available = [
                category for category in compatible if category in self.products_by_category
            ]
            if available:
                resolved.extend(available)
            else:
                missing.append(intent)
        return list(dict.fromkeys(resolved)), missing

    # Chuẩn hóa điểm cá nhân hóa về [0, 1] để có thể kết hợp và giải thích.
    def _personalization_score(self, user_id: str, product_id: str) -> float:
        # Không có user/profile thì dùng mức trung tính; popularity đã có trọng số riêng 5%.
        if not user_id:
            return 0.5
        if self.model and user_id in self.model.user_embeddings:
            return sigmoid(self.model.score(user_id, product_id))
        profile = self.user_profiles.get(user_id)
        vector = self.product_embeddings.get(product_id)
        if profile and vector:
            return _unit_cosine(profile, vector)
        return 0.5

    # Tính mức độ item đáp ứng trực tiếp câu nhu cầu.
    def _need_score(
        self,
        need_tokens: set[str],
        inferred_categories: list[str],
        product: dict,
    ) -> tuple[float, list[str]]:
        category = str(product.get("category", ""))
        category_match = 1.0 if category in inferred_categories else 0.0
        searchable = product_text(product)
        product_tokens = set(tokenize(_ascii_text(searchable)))
        matched = sorted(need_tokens & product_tokens)
        lexical = min(len(matched) / max(min(len(need_tokens), 4), 1), 1.0)
        if inferred_categories:
            return min(0.8 * category_match + 0.2 * lexical, 1.0), matched
        return lexical, matched

    # Xếp hạng Top-K theo nhu cầu hiện tại, lịch sử user và chất lượng sản phẩm.
    def recommend_for_need(
        self,
        need: str,
        user_id: str = "",
        top_k: int = 5,
        prefer_images: bool = True,
        trace: dict | None = None,
    ) -> list[dict]:
        started_at = perf_counter()
        clean_need = need.strip()
        if trace is not None:
            trace.clear()
            trace.update(
                {
                    "query": clean_need,
                    "user_id": user_id,
                    "top_k": top_k,
                    "prefer_images": prefer_images,
                    "catalog_products": len(self.products),
                    "weights": {
                        "need": 0.60,
                        "personalization": 0.20,
                        "quality": 0.15,
                        "popularity": 0.05,
                    },
                }
            )
        if not clean_need:
            results = self.recommend_for_user(user_id, top_k=top_k)
            if trace is not None:
                trace.update(
                    {
                        "status": "fallback_user_recommendation",
                        "returned_count": len(results),
                        "total_ms": round((perf_counter() - started_at) * 1000, 3),
                    }
                )
            return results
        resolve_started_at = perf_counter()
        inferred_categories = self.infer_need_categories(clean_need)
        candidate_categories, missing_categories = self.resolve_need_categories(clean_need)
        need_tokens = {
            token for token in tokenize(_ascii_text(clean_need)) if token not in NEED_QUERY_STOPWORDS
        }
        for token in tuple(need_tokens):
            need_tokens.update(NEED_TOKEN_EXPANSIONS.get(token, set()))
        normalized_need = _ascii_text(clean_need)
        for phrase, expansions in NEED_PHRASE_EXPANSIONS.items():
            if phrase in normalized_need:
                need_tokens.update(expansions)
        if trace is not None:
            trace.update(
                {
                    "inferred_intents": inferred_categories,
                    "resolved_categories": candidate_categories,
                    "missing_categories": missing_categories,
                    "need_tokens": sorted(need_tokens),
                    "resolve_ms": round((perf_counter() - resolve_started_at) * 1000, 3),
                }
            )
        if missing_categories:
            if trace is not None:
                trace.update(
                    {
                        "status": "missing_category",
                        "returned_count": 0,
                        "total_ms": round((perf_counter() - started_at) * 1000, 3),
                        "warnings": ["Nhu cầu thuộc category chưa có trong catalog."],
                    }
                )
            return []
        candidate_ids = [
            product_id
            for category in candidate_categories
            for product_id in self.products_by_category.get(category, [])
        ]
        if not candidate_ids:
            candidate_ids = self.product_ids
        seen = self.seen_by_user.get(user_id, set())
        if trace is not None:
            trace.update(
                {
                    "candidate_count_before_seen_filter": len(candidate_ids),
                    "seen_products_for_user": len(seen),
                }
            )

        # Pha 1 chỉ dùng scalar rẻ để thu hẹp catalog lớn trước khi tokenize/vector scoring.
        phase_1_started_at = perf_counter()
        preliminary = []
        for product_id in candidate_ids:
            if product_id in seen:
                continue
            product = self.products[product_id]
            rating = parse_float(product.get("average_rating"), 0.0)
            quality = max(0.0, min(rating / 5.0, 1.0))
            popularity = math.log1p(parse_int(product.get("rating_number"), 0)) / 10.0
            popularity = min(popularity, 1.0)
            category_match = 1.0 if product.get("category") in candidate_categories else 0.0
            preview = self.product_search_preview.get(product_id, "")
            lexical_hint = min(
                sum(token in preview for token in need_tokens) / max(min(len(need_tokens), 4), 1),
                1.0,
            )
            base_need = (
                0.8 * category_match + 0.2 * lexical_hint
                if inferred_categories
                else lexical_hint
            )
            base_score = 0.60 * base_need + 0.15 * quality + 0.05 * popularity
            preliminary.append((product_id, base_score))
        preliminary.sort(key=lambda row: row[1], reverse=True)
        shortlist_size = max(500, top_k * 100)
        image_shortlist = [
            row for row in preliminary if str(self.products[row[0]].get("image_url", "")).strip()
        ][:shortlist_size]
        other_shortlist = [
            row for row in preliminary if not str(self.products[row[0]].get("image_url", "")).strip()
        ][:shortlist_size]
        shortlist_ids = [product_id for product_id, _ in [*image_shortlist, *other_shortlist]]
        if trace is not None:
            trace.update(
                {
                    "candidate_count_after_seen_filter": len(preliminary),
                    "shortlist_limit_per_image_group": shortlist_size,
                    "shortlist_with_images": len(image_shortlist),
                    "shortlist_without_images": len(other_shortlist),
                    "shortlist_total": len(shortlist_ids),
                    "phase_1_ms": round((perf_counter() - phase_1_started_at) * 1000, 3),
                }
            )

        # Pha 2 mới tính lexical và personalization trên tối đa khoảng 1.000 ứng viên.
        phase_2_started_at = perf_counter()
        scored = []
        for product_id in shortlist_ids:
            product = self.products[product_id]
            need_score, matched = self._need_score(need_tokens, candidate_categories, product)
            personalization = self._personalization_score(user_id, product_id)
            rating = parse_float(product.get("average_rating"), 0.0)
            quality = max(0.0, min(rating / 5.0, 1.0))
            popularity = math.log1p(parse_int(product.get("rating_number"), 0)) / 10.0
            popularity = min(popularity, 1.0)
            total = 0.60 * need_score + 0.20 * personalization + 0.15 * quality + 0.05 * popularity
            scored.append((product_id, total, need_score, personalization, quality, popularity, matched))
        scored.sort(key=lambda row: row[1], reverse=True)
        if prefer_images:
            with_images = [
                row for row in scored if str(self.products[row[0]].get("image_url", "")).strip()
            ]
            without_images = [
                row for row in scored if not str(self.products[row[0]].get("image_url", "")).strip()
            ]
            scored = [*with_images, *without_images]
        if trace is not None:
            trace.update(
                {
                    "scored_count": len(scored),
                    "phase_2_ms": round((perf_counter() - phase_2_started_at) * 1000, 3),
                }
            )

        results = []
        for product_id, total, need_score, personalization, quality, popularity, matched in scored[:top_k]:
            item = self._enrich(product_id, total, user_id)
            category = str(item.get("category", ""))
            category_label = CATEGORY_LABELS.get(category, category.replace("_", " "))
            reasons = [f"thuộc nhóm {category_label}"] if category in candidate_categories else []
            if matched:
                reasons.append(f"khớp từ khóa: {', '.join(matched[:4])}")
            reasons.append(f"đánh giá trung bình {parse_float(item.get('average_rating'), 0.0):.1f}/5")
            if user_id:
                reasons.append(f"điểm cá nhân hóa {personalization * 100:.0f}%")
            item["match_percentage"] = round(total * 100, 1)
            item["score_breakdown"] = {
                "Nhu cầu": round(need_score * 100, 1),
                "Cá nhân hóa": round(personalization * 100, 1),
                "Chất lượng": round(quality * 100, 1),
                "Phổ biến": round(popularity * 100, 1),
            }
            item["matched_need_tokens"] = matched
            item["explanation"] = "Được chọn vì " + "; ".join(reasons) + "."
            results.append(item)
        if trace is not None:
            result_ids = [item["product_id"] for item in results]
            trace.update(
                {
                    "status": "ok" if results else "no_candidates",
                    "returned_count": len(results),
                    "unique_returned_count": len(set(result_ids)),
                    "seen_item_leakage_count": len(set(result_ids) & seen),
                    "image_result_count": sum(bool(item.get("image_url")) for item in results),
                    "top_results": [
                        {
                            "rank": rank,
                            "product_id": item["product_id"],
                            "title": item.get("title", ""),
                            "category": item.get("category", ""),
                            "score": round(float(item.get("score", 0.0)), 6),
                            "score_breakdown": item.get("score_breakdown", {}),
                            "matched_need_tokens": item.get("matched_need_tokens", []),
                        }
                        for rank, item in enumerate(results, start=1)
                    ],
                    "total_ms": round((perf_counter() - started_at) * 1000, 3),
                }
            )
            semantic_matches = sum(bool(item.get("matched_need_tokens")) for item in results)
            trace["semantic_match_result_count"] = semantic_matches
            trace["semantic_match_result_rate"] = round(
                semantic_matches / max(len(results), 1), 6
            )
        return results

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
        target = self.modality_embeddings.get(product_id)
        target_product = self.products.get(product_id, {})
        if not target:
            return self._similar_products_lightweight(product_id, top_k)
        category = target_product.get("category")
        candidate_ids = self.products_by_category.get(str(category), self.product_ids)
        rows = []
        for other_id in candidate_ids:
            if other_id == product_id or other_id not in self.modality_embeddings:
                continue
            other = self.modality_embeddings[other_id]
            text_score = _unit_cosine(target["text"], other["text"])
            image_score = _unit_cosine(target["image"], other["image"])
            metadata_score = _unit_cosine(target["metadata"], other["metadata"])
            fused_score = _unit_cosine(target["fused"], other["fused"])
            score = 0.40 * text_score + 0.15 * image_score + 0.25 * metadata_score + 0.20 * fused_score
            rows.append(
                (other_id, score, text_score, image_score, metadata_score, fused_score)
            )
        rows.sort(key=lambda row: row[1], reverse=True)
        with_images = [
            row for row in rows if str(self.products[row[0]].get("image_url", "")).strip()
        ]
        without_images = [
            row for row in rows if not str(self.products[row[0]].get("image_url", "")).strip()
        ]
        rows = [*with_images, *without_images]

        results = []
        for other_id, score, text_score, image_score, metadata_score, fused_score in rows[:top_k]:
            item = self._enrich(other_id, score, "")
            item["match_percentage"] = round(score * 100, 1)
            item["score_breakdown"] = {
                "Text": round(text_score * 100, 1),
                "Image signal": round(image_score * 100, 1),
                "Metadata": round(metadata_score * 100, 1),
                "Fused": round(fused_score * 100, 1),
            }
            item["explanation"] = (
                f"Cùng nhóm công năng {str(category).replace('_', ' ')}; "
                f"text gần {text_score * 100:.0f}%, metadata {metadata_score * 100:.0f}% "
                f"và tín hiệu ảnh {image_score * 100:.0f}%."
            )
            results.append(item)
        return results

    # Fallback nhanh cho app: không load file embedding JSONL hàng trăm MB vào RAM.
    def _similar_products_lightweight(self, product_id: str, top_k: int) -> list[dict]:
        target = self.products.get(product_id)
        if not target:
            return []
        category = str(target.get("category", ""))
        target_tokens = set(
            tokenize(
                " ".join(
                    [
                        str(target.get("title", "")),
                        str(target.get("features", "")),
                        category,
                    ]
                )
            )
        )
        target_price = parse_float(target.get("price"), 0.0)
        target_rating = parse_float(target.get("average_rating"), 0.0)
        rows = []
        for other_id in self.products_by_category.get(category, self.product_ids):
            if other_id == product_id:
                continue
            other = self.products[other_id]
            other_tokens = set(
                tokenize(
                    " ".join(
                        [
                            str(other.get("title", "")),
                            str(other.get("features", "")),
                            category,
                        ]
                    )
                )
            )
            union = target_tokens | other_tokens
            text_score = len(target_tokens & other_tokens) / max(len(union), 1)
            other_price = parse_float(other.get("price"), 0.0)
            price_score = 1.0 - min(abs(target_price - other_price) / max(target_price, 1.0), 1.0)
            rating_score = 1.0 - min(
                abs(target_rating - parse_float(other.get("average_rating"), 0.0)) / 5.0,
                1.0,
            )
            metadata_score = 0.6 * price_score + 0.4 * rating_score
            image_score = 1.0 if target.get("image_url") and other.get("image_url") else 0.0
            fused_score = 0.55 * text_score + 0.35 * metadata_score + 0.10 * image_score
            rows.append((other_id, fused_score, text_score, image_score, metadata_score))
        rows.sort(
            key=lambda row: (
                bool(str(self.products[row[0]].get("image_url", "")).strip()),
                row[1],
            ),
            reverse=True,
        )
        results = []
        for other_id, score, text_score, image_score, metadata_score in rows[:top_k]:
            item = self._enrich(other_id, score, "")
            item["match_percentage"] = round(score * 100, 1)
            item["score_breakdown"] = {
                "Text": round(text_score * 100, 1),
                "Image availability": round(image_score * 100, 1),
                "Metadata": round(metadata_score * 100, 1),
                "Fused": round(score * 100, 1),
            }
            item["explanation"] = (
                f"Cùng nhóm công năng {category.replace('_', ' ')}; "
                f"text gần {text_score * 100:.0f}% và metadata {metadata_score * 100:.0f}%."
            )
            results.append(item)
        return results

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
