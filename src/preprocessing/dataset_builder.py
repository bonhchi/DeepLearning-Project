# Tạo các bảng phục vụ recommendation từ review Amazon Reviews 2023.

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from src.data.amazon_reviews import iter_reviews
from src.io_utils import parse_float, parse_int, parse_bool, write_csv_rows
from src.vector_ops import stable_hash_int


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


# Chia train/validation/test theo chiến lược giữ lại tương tác cuối của mỗi user.
def split_interactions(interactions: list[dict]) -> list[dict]:
    positive_by_user: dict[str, list[dict]] = defaultdict(list)
    for row in interactions:
        if parse_int(row.get("label"), 0) == 1:
            positive_by_user[row["user_id"]].append(row)
        row["split"] = "train"
    for rows in positive_by_user.values():
        rows.sort(key=lambda item: parse_int(item.get("timestamp"), 0))
        if len(rows) >= 2:
            rows[-1]["split"] = "test"
        if len(rows) >= 3:
            rows[-2]["split"] = "val"
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
def build_products(reviews: list[dict]) -> list[dict]:
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
        stats["rating_sum"] += parse_float(review.get("rating"), 0.0)
        stats["rating_count"] += 1
        review_text = f"{review.get('review_title', '')} {review.get('review_text', '')}".strip()
        if len(stats["text_parts"]) < 20:
            stats["text_parts"].append(review_text)
        if len(stats["description_parts"]) < 3 and review.get("review_text"):
            stats["description_parts"].append(str(review["review_text"]))
        if review.get("source_category"):
            stats["category_counts"][str(review["source_category"])] += 1
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
    for user_id, rows in grouped.items():
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
    products = build_products(reviews)
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
        [
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
        ],
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
        [
            "user_id",
            "product_id",
            "review_title",
            "review_text",
            "rating",
            "timestamp",
            "helpful_vote",
            "source_category",
        ],
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
