# Tạo các bảng phục vụ recommendation từ review Amazon Reviews 2023.

from __future__ import annotations

import math
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from src.data.amazon_reviews import iter_reviews
from src.io_utils import (
    json_loads_safe,
    parse_float,
    parse_int,
    parse_bool,
    read_csv_rows,
    write_csv_rows,
)
from src.vector_ops import stable_hash_int


REVIEW_TRAIN_METADATA_SOURCE = "review_train_aggregate"
AMAZON_ITEM_METADATA_SOURCE = "amazon_item_metadata"
AMAZON_METADATA_FIELDS = (
    "title",
    "store",
    "price",
    "average_rating",
    "rating_number",
    "description",
    "features",
    "image_url",
    "preferred_price_range",
)
PRODUCT_FIELDS = [
    "product_id",
    "title",
    "category",
    "store",
    "price",
    "average_rating",
    "rating_number",
    "description",
    "features",
    "image_url",
    "preferred_price_range",
    "metadata_source",
    "metadata_provenance",
]

REVIEW_FIELDS = [
    "user_id",
    "product_id",
    "review_title",
    "review_text",
    "rating",
    "timestamp",
    "helpful_vote",
    "verified_purchase",
    "image_url",
    "source_category",
]
DATASET_MANIFEST_VERSION = 1


def write_dataset_manifest(output_dir: str | Path) -> Path:
    """Certify the temporal/provenance rules used by the processed tables."""

    target = Path(output_dir) / "dataset_manifest.json"
    payload = {
        "artifact_version": DATASET_MANIFEST_VERSION,
        "leakage_safe": True,
        "interaction_split": "per_user_chronological_cutoff",
        "product_review_features": "train_events_only",
        "user_profiles": "train_events_only",
        "review_join": "user_product_exact_raw_timestamp",
        "metadata_provenance": "field_level",
        "business_inventory": "deterministic_demo_proxy_not_live_stock",
    }
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


def parse_metadata_provenance(product: dict) -> dict[str, str]:
    """Return field-level provenance from either in-memory or CSV products.

    Legacy catalogs have neither provenance column.  Treating those fields as
    unverified is deliberately conservative: append will rebuild them from the
    newly computed train split instead of preserving potentially held-out text.
    """

    value = json_loads_safe(product.get("metadata_provenance"), {})
    if not isinstance(value, dict):
        return {}
    return {
        str(field): str(source)
        for field, source in value.items()
        if str(field) and str(source)
    }


def overlay_verified_amazon_metadata(rebuilt: dict, existing: dict | None) -> dict:
    """Restore only fields explicitly proven to come from Amazon item metadata."""

    if not existing:
        return rebuilt
    existing_provenance = parse_metadata_provenance(existing)
    restored_fields: list[str] = []
    for field in AMAZON_METADATA_FIELDS:
        if existing_provenance.get(field) != AMAZON_ITEM_METADATA_SOURCE:
            continue
        value = existing.get(field)
        if value in (None, ""):
            continue
        if field == "features":
            value = json_loads_safe(value, value)
        rebuilt[field] = value
        restored_fields.append(field)

    if restored_fields:
        provenance = parse_metadata_provenance(rebuilt)
        provenance.update({field: AMAZON_ITEM_METADATA_SOURCE for field in restored_fields})
        rebuilt["metadata_source"] = AMAZON_ITEM_METADATA_SOURCE
        rebuilt["metadata_provenance"] = provenance
    return rebuilt


# Chuyển timestamp mili-giây thành bucket ngày dạng số nguyên.
def timestamp_day(timestamp_ms: int) -> int:
    if timestamp_ms <= 0:
        return 0
    return timestamp_ms // 86_400_000


# Ánh xạ rating, xác thực mua hàng và helpfulness thành trọng số event.
def event_weight(review: dict) -> float:
    rating = parse_float(review.get("rating"), 0.0)
    weight = max(0.1, rating)
    if parse_bool(review.get("verified_purchase")):
        weight += 0.5
    if parse_int(review.get("helpful_vote"), 0) >= 3:
        weight += 0.2
    return round(min(weight, 5.0), 3)


# Gán loại hành vi dựa trên rating và trạng thái mua hàng xác thực.
def event_type(review: dict) -> str:
    rating = parse_float(review.get("rating"), 0.0)
    verified = parse_bool(review.get("verified_purchase"))
    if rating >= 4.0 and verified:
        return "verified_positive_review"
    if rating >= 4.0:
        return "positive_review"
    if rating >= 3.0:
        return "weak_review"
    return "negative_review"


# Suy luận category thời trang cấp thô từ text sản phẩm và review.
def infer_category(text: str) -> str:
    lowered = text.lower()
    rules = [
        ("shoe", "shoes"),
        ("sneaker", "shoes"),
        ("boot", "shoes"),
        ("sock", "socks"),
        ("dress", "dresses"),
        ("shirt", "tops"),
        ("blouse", "tops"),
        ("jacket", "outerwear"),
        ("coat", "outerwear"),
        ("pant", "bottoms"),
        ("jean", "bottoms"),
        ("bag", "bags"),
        ("purse", "bags"),
        ("watch", "watches"),
        ("glasses", "eyewear"),
        ("sunglasses", "eyewear"),
        ("necklace", "jewelry"),
        ("ring", "jewelry"),
        ("earring", "jewelry"),
        ("bracelet", "jewelry"),
    ]
    for keyword, category in rules:
        if keyword in lowered:
            return category
    return "fashion_accessories"


# Tạo giá bán giả lập ổn định để thử nghiệm metadata.
def synthetic_price(product_id: str, category: str) -> float:
    category_ranges = {
        "jewelry": (8, 120),
        "shoes": (25, 180),
        "socks": (5, 35),
        "dresses": (20, 160),
        "tops": (10, 90),
        "outerwear": (35, 240),
        "bottoms": (18, 120),
        "bags": (18, 220),
        "watches": (20, 260),
        "eyewear": (12, 140),
        "fashion_accessories": (6, 100),
    }
    low, high = category_ranges.get(category, (10, 120))
    fraction = stable_hash_int(product_id, 10_000) / 10_000
    return round(low + (high - low) * fraction, 2)


# Ánh xạ giá số sang bucket giá dễ hiểu cho khách hàng.
def price_bucket(price: float) -> str:
    if price < 20:
        return "budget"
    if price < 60:
        return "mid_range"
    if price < 140:
        return "premium"
    return "luxury"


# Tạo session id ổn định từ user và timestamp theo ngày.
def session_id_for(review: dict) -> str:
    user_id = str(review.get("user_id", "unknown_user"))
    day = timestamp_day(parse_int(review.get("timestamp"), 0))
    return f"s_{stable_hash_int(user_id + str(day), 10**14)}"


# Chia theo timeline user; mọi event sau cutoff holdout đều không thể quay lại train.
def split_interactions(interactions: list[dict]) -> list[dict]:
    rows_by_user: dict[str, list[dict]] = defaultdict(list)
    for row in interactions:
        row["split"] = "train"
        rows_by_user[row["user_id"]].append(row)
    for user_rows in rows_by_user.values():
        positive_timestamps = sorted(
            parse_int(row.get("timestamp"), 0)
            for row in user_rows
            if parse_int(row.get("label"), 0) == 1
        )
        if len(positive_timestamps) < 2:
            continue
        test_cutoff = positive_timestamps[-1]
        validation_cutoff = (
            positive_timestamps[-2] if len(positive_timestamps) >= 3 else None
        )
        for row in user_rows:
            timestamp = parse_int(row.get("timestamp"), 0)
            if timestamp >= test_cutoff:
                row["split"] = "test"
            elif validation_cutoff is not None and timestamp >= validation_cutoff:
                row["split"] = "val"
    return interactions


# Tạo bảng interactions từ review đã chuẩn hóa.
def build_interactions(reviews: list[dict]) -> list[dict]:
    rows = []
    for review in reviews:
        rating = parse_float(review.get("rating"), 0.0)
        rows.append(
            {
                "user_id": review["user_id"],
                "product_id": review["product_id"],
                "event_type": event_type(review),
                "event_weight": event_weight(review),
                "rating": rating,
                "timestamp": parse_int(review.get("timestamp"), 0),
                "verified_purchase": parse_bool(review.get("verified_purchase")),
                "session_id": session_id_for(review),
                "label": 1 if rating >= 4.0 else 0,
                "split": "train",
                "source_category": review.get("source_category", ""),
            }
        )
    return split_interactions(rows)


# Trích xuất keyword nhẹ từ nội dung review.
def top_keywords(text: str, limit: int = 8) -> list[str]:
    words = []
    for token in text.lower().replace("/", " ").replace("-", " ").split():
        cleaned = "".join(char for char in token if char.isalnum())
        if len(cleaned) > 3:
            words.append(cleaned)
    stopwords = {"this", "that", "with", "have", "they", "were", "from", "very", "just", "great"}
    counts = Counter(word for word in words if word not in stopwords)
    return [word for word, _ in counts.most_common(limit)]


# Gom review để tạo bảng metadata sản phẩm gọn trong bộ nhớ.
def build_products(
    reviews: list[dict],
    training_review_keys: set[tuple[str, str, int]] | None = None,
) -> list[dict]:
    """Build catalog features without reading held-out review content.

    Product identity and source category are catalog-level fields and may use all
    rows. Rating aggregates, title, description, keywords and review images only
    use rows whose exact event key belongs to the training split when
    ``training_review_keys`` is provided.
    """

    grouped: dict[str, dict] = {}
    for review in reviews:
        product_id = review["product_id"]
        stats = grouped.setdefault(
            product_id,
            {
                "rating_sum": 0.0,
                "rating_count": 0,
                "text_parts": [],
                "description_parts": [],
                "category_counts": Counter(),
                "image_url": "",
                "title": "",
            },
        )
        if review.get("source_category"):
            stats["category_counts"][str(review["source_category"])] += 1
        review_key = (
            str(review.get("user_id", "")),
            str(review.get("product_id", "")),
            parse_int(review.get("timestamp"), 0),
        )
        if training_review_keys is not None and review_key not in training_review_keys:
            continue
        stats["rating_sum"] += parse_float(review.get("rating"), 0.0)
        stats["rating_count"] += 1
        review_text = f"{review.get('review_title', '')} {review.get('review_text', '')}".strip()
        if len(stats["text_parts"]) < 20:
            stats["text_parts"].append(review_text)
        if len(stats["description_parts"]) < 3 and review.get("review_text"):
            stats["description_parts"].append(str(review["review_text"]))
        if not stats["image_url"] and review.get("image_url"):
            stats["image_url"] = str(review["image_url"])
        if not stats["title"] and review.get("review_title"):
            stats["title"] = str(review["review_title"])

    products = []
    for product_id, stats in grouped.items():
        text_blob = " ".join(stats["text_parts"])
        category = (
            stats["category_counts"].most_common(1)[0][0]
            if stats["category_counts"]
            else infer_category(text_blob)
        )
        price = synthetic_price(product_id, category)
        representative_title = stats["title"]
        title = representative_title if len(representative_title) > 3 else f"Amazon Item {product_id}"
        store = f"store_{stable_hash_int(product_id, 30) + 1:02d}"
        products.append(
            {
                "product_id": product_id,
                "title": title,
                "category": category,
                "store": store,
                "price": price,
                "average_rating": round(
                    stats["rating_sum"] / max(stats["rating_count"], 1), 3
                ),
                "rating_number": stats["rating_count"],
                "description": " ".join(stats["description_parts"])[:600],
                "features": top_keywords(text_blob),
                "image_url": stats["image_url"],
                "preferred_price_range": price_bucket(price),
                "metadata_source": REVIEW_TRAIN_METADATA_SOURCE,
                "metadata_provenance": {
                    "title": REVIEW_TRAIN_METADATA_SOURCE,
                    "category": "review_source_category",
                    "store": "synthetic_catalog",
                    "price": "synthetic_catalog",
                    "average_rating": REVIEW_TRAIN_METADATA_SOURCE,
                    "rating_number": REVIEW_TRAIN_METADATA_SOURCE,
                    "description": REVIEW_TRAIN_METADATA_SOURCE,
                    "features": REVIEW_TRAIN_METADATA_SOURCE,
                    "image_url": REVIEW_TRAIN_METADATA_SOURCE,
                    "preferred_price_range": "synthetic_catalog",
                },
            }
        )
    products.sort(key=lambda row: row["product_id"])
    return products


# Tổng hợp tương tác để tạo hồ sơ và sở thích người dùng.
def build_users(interactions: list[dict], products: list[dict]) -> list[dict]:
    product_map = {row["product_id"]: row for row in products}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in interactions:
        grouped[row["user_id"]].append(row)

    users = []
    for user_id, all_rows in grouped.items():
        rows = [row for row in all_rows if row.get("split", "train") == "train"]
        if not rows:
            continue
        ratings = [parse_float(row.get("rating"), 0.0) for row in rows]
        positive_rows = [row for row in rows if parse_int(row.get("label"), 0) == 1]
        category_counts = Counter(product_map.get(row["product_id"], {}).get("category", "unknown") for row in positive_rows)
        store_counts = Counter(product_map.get(row["product_id"], {}).get("store", "unknown") for row in positive_rows)
        prices = [parse_float(product_map.get(row["product_id"], {}).get("price"), 0.0) for row in positive_rows]
        activity_level = "high" if len(rows) >= 10 else "medium" if len(rows) >= 3 else "low"
        avg_rating = sum(ratings) / max(len(ratings), 1)
        if activity_level == "high" and avg_rating >= 4.0:
            segment = "loyal_high_value"
        elif avg_rating < 3.0:
            segment = "needs_recovery"
        elif len(category_counts) >= 3:
            segment = "style_explorer"
        else:
            segment = "casual_browser"
        avg_price = sum(prices) / max(len(prices), 1) if prices else 0.0
        users.append(
            {
                "user_id": user_id,
                "user_segment": segment,
                "activity_level": activity_level,
                "avg_rating": round(avg_rating, 3),
                "preferred_categories": [category for category, _ in category_counts.most_common(3)],
                "preferred_price_range": price_bucket(avg_price) if avg_price else "unknown",
                "preferred_store": store_counts.most_common(1)[0][0] if store_counts else "unknown",
            }
        )
    users.sort(key=lambda row: row["user_id"])
    return users


# Tạo bảng reviews theo proposal với các cột cần thiết.
def build_reviews_table(reviews: list[dict]) -> list[dict]:
    return [
        {
            "user_id": row["user_id"],
            "product_id": row["product_id"],
            "review_title": row.get("review_title", ""),
            "review_text": row.get("review_text", ""),
            "rating": row.get("rating", 0.0),
            "timestamp": row.get("timestamp", 0),
            "helpful_vote": row.get("helpful_vote", 0),
            "verified_purchase": row.get("verified_purchase", False),
            "image_url": row.get("image_url", ""),
            "source_category": row.get("source_category", ""),
        }
        for row in reviews
    ]


# Gom các sự kiện tương tác thành phiên mua sắm giả lập.
def build_sessions(interactions: list[dict], products: list[dict]) -> list[dict]:
    product_map = {row["product_id"]: row for row in products}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in interactions:
        grouped[row["session_id"]].append(row)

    sessions = []
    devices = ["mobile", "desktop", "tablet"]
    for session_id, rows in grouped.items():
        rows.sort(key=lambda item: parse_int(item.get("timestamp"), 0))
        categories = Counter(product_map.get(row["product_id"], {}).get("category", "fashion") for row in rows)
        top_category = categories.most_common(1)[0][0]
        sessions.append(
            {
                "session_id": session_id,
                "user_id": rows[0]["user_id"],
                "timestamp": rows[0].get("timestamp", 0),
                "session_intent": f"browse_{top_category}",
                "device": devices[stable_hash_int(session_id, len(devices))],
                "event_sequence": [f"{row['event_type']}:{row['product_id']}" for row in rows],
            }
        )
    sessions.sort(key=lambda row: row["session_id"])
    return sessions


# Tạo tín hiệu business context ổn định cho thử nghiệm ranking.
def build_business_context(products: list[dict]) -> list[dict]:
    rows = []
    for product in products:
        product_id = product["product_id"]
        seed = stable_hash_int(product_id, 1_000_000) / 1_000_000
        discount = round((stable_hash_int(product_id + "discount", 35) / 100), 2)
        risk = round(stable_hash_int(product_id + "risk", 100) / 100, 2)
        inventory = round(0.25 + 0.75 * seed, 3)
        margin = round(0.15 + 0.75 * (stable_hash_int(product_id + "margin", 10_000) / 10_000), 3)
        rows.append(
            {
                "product_id": product_id,
                "inventory_score": inventory,
                "margin_score": margin,
                "discount_rate": discount,
                "risk_score": risk,
                "campaign_eligible": discount >= 0.15 and risk <= 0.65,
            }
        )
    return rows


# Tạo và ghi toàn bộ bảng CSV từ một iterable review đã chuẩn hóa.
def write_processed_dataset_from_reviews(reviews_source: Iterable[dict], output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reviews = list(reviews_source)
    interactions = build_interactions(reviews)
    training_review_keys = {
        (
            str(row.get("user_id", "")),
            str(row.get("product_id", "")),
            parse_int(row.get("timestamp"), 0),
        )
        for row in interactions
        if row.get("split", "train") == "train"
    }
    products = build_products(reviews, training_review_keys)
    users = build_users(interactions, products)
    sessions = build_sessions(interactions, products)
    business_context = build_business_context(products)

    write_csv_rows(
        output / "interactions.csv",
        interactions,
        [
            "user_id",
            "product_id",
            "event_type",
            "event_weight",
            "rating",
            "timestamp",
            "verified_purchase",
            "session_id",
            "label",
            "split",
            "source_category",
        ],
    )
    write_csv_rows(
        output / "products.csv",
        products,
        PRODUCT_FIELDS,
    )
    write_csv_rows(
        output / "users.csv",
        users,
        [
            "user_id",
            "user_segment",
            "activity_level",
            "avg_rating",
            "preferred_categories",
            "preferred_price_range",
            "preferred_store",
        ],
    )
    write_csv_rows(
        output / "reviews.csv",
        build_reviews_table(reviews),
        REVIEW_FIELDS,
    )
    write_csv_rows(
        output / "sessions.csv",
        sessions,
        ["session_id", "user_id", "timestamp", "session_intent", "device", "event_sequence"],
    )
    write_csv_rows(
        output / "business_context.csv",
        business_context,
        ["product_id", "inventory_score", "margin_score", "discount_rate", "risk_score", "campaign_eligible"],
    )
    write_dataset_manifest(output)
    return {
        "reviews": len(reviews),
        "reviews_by_category": dict(
            sorted(Counter(row.get("source_category") or "local" for row in reviews).items())
        ),
        "users": len(users),
        "products": len(products),
        "interactions": len(interactions),
        "sessions": len(sessions),
        "business_context_rows": len(business_context),
    }


# Giữ API cũ: đọc JSONL local rồi chuyển sang pipeline dùng chung.
def write_processed_dataset(
    raw_path: str | Path,
    output_dir: str | Path,
    limit: int | None = 10000,
) -> dict:
    return write_processed_dataset_from_reviews(
        iter_reviews(raw_path, limit=limit),
        output_dir,
    )


def append_processed_dataset_from_reviews(
    reviews_source: Iterable[dict],
    output_dir: str | Path,
) -> dict:
    """Ghép review mới vào processed catalog mà vẫn giữ metadata và ảnh hiện có."""
    output = Path(output_dir)
    products_path = output / "products.csv"
    interactions_path = output / "interactions.csv"
    reviews_path = output / "reviews.csv"
    if not products_path.exists() or not interactions_path.exists() or not reviews_path.exists():
        return write_processed_dataset_from_reviews(reviews_source, output)

    existing_products = read_csv_rows(products_path)
    existing_interactions = read_csv_rows(interactions_path)
    existing_reviews = read_csv_rows(reviews_path)
    existing_review_keys = {
        (
            row.get("user_id", ""),
            row.get("product_id", ""),
            parse_int(row.get("timestamp"), 0),
        )
        for row in existing_reviews
    }
    new_reviews = []
    duplicates_skipped = 0
    for review in reviews_source:
        key = (
            str(review.get("user_id", "")),
            str(review.get("product_id", "")),
            parse_int(review.get("timestamp"), 0),
        )
        if key in existing_review_keys:
            duplicates_skipped += 1
            continue
        existing_review_keys.add(key)
        new_reviews.append(review)

    if not new_reviews:
        return {
            "reviews": len(existing_reviews),
            "products": len(existing_products),
            "interactions": len(existing_interactions),
            "appended_reviews": 0,
            "duplicate_reviews_skipped": duplicates_skipped,
        }

    interactions = split_interactions([*existing_interactions, *build_interactions(new_reviews)])
    training_review_keys = {
        (
            str(row.get("user_id", "")),
            str(row.get("product_id", "")),
            parse_int(row.get("timestamp"), 0),
        )
        for row in interactions
        if row.get("split", "train") == "train"
    }
    reviews_table = [*existing_reviews, *build_reviews_table(new_reviews)]
    existing_product_map = {row["product_id"]: row for row in existing_products}
    # Rebuild every review-derived field against the newly computed global
    # train split.  This matters when appending an older event moves a formerly
    # training event into validation/test.  Only catalog metadata with explicit
    # field-level provenance is restored afterward.
    products = [
        overlay_verified_amazon_metadata(
            product_row,
            existing_product_map.get(product_row["product_id"]),
        )
        for product_row in build_products(reviews_table, training_review_keys)
    ]
    users = build_users(interactions, products)
    sessions = build_sessions(interactions, products)
    business_context = build_business_context(products)

    write_csv_rows(
        interactions_path,
        interactions,
        [
            "user_id", "product_id", "event_type", "event_weight", "rating", "timestamp",
            "verified_purchase", "session_id", "label", "split", "source_category",
        ],
    )
    write_csv_rows(products_path, products, PRODUCT_FIELDS)
    write_csv_rows(
        output / "users.csv",
        users,
        [
            "user_id", "user_segment", "activity_level", "avg_rating",
            "preferred_categories", "preferred_price_range", "preferred_store",
        ],
    )
    write_csv_rows(
        reviews_path,
        reviews_table,
        REVIEW_FIELDS,
    )
    write_csv_rows(
        output / "sessions.csv",
        sessions,
        ["session_id", "user_id", "timestamp", "session_intent", "device", "event_sequence"],
    )
    write_csv_rows(
        output / "business_context.csv",
        business_context,
        [
            "product_id", "inventory_score", "margin_score", "discount_rate",
            "risk_score", "campaign_eligible",
        ],
    )
    write_dataset_manifest(output)
    return {
        "reviews": len(reviews_table),
        "users": len(users),
        "products": len(products),
        "interactions": len(interactions),
        "sessions": len(sessions),
        "business_context_rows": len(business_context),
        "appended_reviews": len(new_reviews),
        "duplicate_reviews_skipped": duplicates_skipped,
    }
