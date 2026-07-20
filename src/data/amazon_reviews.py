# Bộ đọc dữ liệu Amazon Reviews 2023 từ JSONL local hoặc Hugging Face.

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.io_utils import parse_float, parse_int, parse_bool, read_csv_rows
from src.vector_ops import stable_hash_int


HF_REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
HF_REVISION = "main"
DEFAULT_CATEGORIES = (
    "Automotive",
    "Electronics",
    "Health_and_Household",
    "Beauty_and_Personal_Care",
)
SUPPORTED_CATEGORIES = (*DEFAULT_CATEGORIES, "Appliances")
METADATA_CATEGORIES = (*SUPPORTED_CATEGORIES, "Amazon_Fashion")


# Chọn URL ảnh tốt nhất có trong payload ảnh của review.
def extract_image_url(images: object) -> str:
    if not images or not isinstance(images, list):
        return ""
    first = images[0]
    if isinstance(first, dict):
        for key in ["large_image_url", "medium_image_url", "small_image_url", "attachment_url"]:
            if first.get(key):
                return str(first[key])
    if isinstance(first, str):
        return first
    return ""


# Chuyển một review Amazon thô sang schema của dự án.
def normalize_review(raw: dict, category: str = "") -> dict:
    user_id = str(raw.get("user_id") or "unknown_user")
    product_id = str(raw.get("parent_asin") or raw.get("asin") or "unknown_product")
    timestamp = parse_int(raw.get("timestamp"), 0)
    review_id = f"r_{stable_hash_int(user_id + product_id + str(timestamp), 10**16)}"
    return {
        "review_id": review_id,
        "user_id": user_id,
        "product_id": product_id,
        "asin": str(raw.get("asin") or product_id),
        "rating": parse_float(raw.get("rating"), 0.0),
        "review_title": str(raw.get("title") or ""),
        "review_text": str(raw.get("text") or ""),
        "timestamp": timestamp,
        "helpful_vote": parse_int(raw.get("helpful_vote"), 0),
        "verified_purchase": parse_bool(raw.get("verified_purchase")),
        "image_url": extract_image_url(raw.get("images")),
        "source_category": category or str(raw.get("source_category") or ""),
    }


# Đọc streaming review đã chuẩn hóa từ file Amazon JSONL.
def iter_reviews(path: str | Path, limit: int | None = None) -> Iterator[dict]:
    yielded = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and yielded >= limit:
                break
            if not line.strip():
                continue
            try:
                yield normalize_review(json.loads(line))
                yielded += 1
            except json.JSONDecodeError:
                continue


# Đọc JSONL HTTP tuần tự bằng standard library, không cần datasets/pandas/numpy.
def iter_remote_jsonl(url: str, timeout: int = 120) -> Iterator[dict]:
    request = Request(
        url,
        headers={
            "Accept": "application/json, application/octet-stream",
            "User-Agent": "amazon-reviews-local-demo/1.0",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                try:
                    yield json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Cannot stream Hugging Face JSONL from {url}: {exc}") from exc


# Tạo URL resolve trực tiếp tới JSONL trên Hugging Face Hub.
def huggingface_review_url(category: str, revision: str = HF_REVISION) -> str:
    return (
        f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/{revision}/"
        f"raw/review_categories/{category}.jsonl"
    )


# Tạo URL tới metadata catalog chứa title, store, price và ảnh sản phẩm chính.
def huggingface_metadata_url(category: str, revision: str = HF_REVISION) -> str:
    return (
        f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/{revision}/"
        f"raw/meta_categories/meta_{category}.jsonl"
    )


# Chọn ảnh catalog MAIN tốt nhất từ schema JSONL hoặc schema dict-of-lists.
def extract_catalog_image_url(images: object) -> str:
    if not images:
        return ""
    if isinstance(images, list):
        candidates = [item for item in images if isinstance(item, dict)]
        candidates.sort(key=lambda item: item.get("variant") != "MAIN")
        for image in candidates:
            for key in ("hi_res", "large", "thumb"):
                if image.get(key):
                    return str(image[key])
    if isinstance(images, dict):
        variants = images.get("variant") or []
        preferred_indexes = list(range(max((len(value) for value in images.values() if isinstance(value, list)), default=0)))
        if "MAIN" in variants:
            main_index = variants.index("MAIN")
            preferred_indexes = [main_index, *[index for index in preferred_indexes if index != main_index]]
        for index in preferred_indexes:
            for key in ("hi_res", "large", "thumb"):
                values = images.get(key) or []
                if index < len(values) and values[index]:
                    return str(values[index])
    return ""


# Chuẩn hóa metadata thật để ghép với products.csv bằng parent_asin.
def normalize_product_metadata(raw: dict, source_category: str = "") -> dict:
    description = raw.get("description") or []
    features = raw.get("features") or []
    return {
        "product_id": str(raw.get("parent_asin") or ""),
        "title": str(raw.get("title") or ""),
        "store": str(raw.get("store") or ""),
        "price": parse_float(raw.get("price"), 0.0),
        "average_rating": parse_float(raw.get("average_rating"), 0.0),
        "rating_number": parse_int(raw.get("rating_number"), 0),
        "description": " ".join(str(item) for item in description) if isinstance(description, list) else str(description),
        "features": list(features) if isinstance(features, list) else [],
        "image_url": extract_catalog_image_url(raw.get("images")),
        "main_category": str(raw.get("main_category") or source_category),
        "source_category": source_category,
    }


# Kiểm tra và loại category trùng nhưng vẫn giữ nguyên thứ tự đầu vào.
def validate_categories(categories: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(category).strip() for category in categories))
    if not selected:
        raise ValueError("At least one Amazon Reviews category is required.")
    invalid = [category for category in selected if category not in SUPPORTED_CATEGORIES]
    if invalid:
        allowed = ", ".join(SUPPORTED_CATEGORIES)
        raise ValueError(f"Unsupported categories: {', '.join(invalid)}. Allowed: {allowed}")
    return selected


def validate_metadata_categories(categories: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(category).strip() for category in categories))
    if not selected:
        raise ValueError("At least one metadata category is required.")
    invalid = [category for category in selected if category not in METADATA_CATEGORIES]
    if invalid:
        allowed = ", ".join(METADATA_CATEGORIES)
        raise ValueError(f"Unsupported metadata categories: {', '.join(invalid)}. Allowed: {allowed}")
    return selected


# Đọc tuần tự từng category qua Hugging Face mà không tải toàn bộ file.
def iter_huggingface_reviews(
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    limit_per_category: int | None = 100_000,
    revision: str = HF_REVISION,
    progress_callback: Callable[[str], None] | None = None,
    progress_every: int = 10_000,
) -> Iterator[dict]:
    if limit_per_category is not None and limit_per_category <= 0:
        raise ValueError("limit_per_category must be greater than zero or None.")
    selected = validate_categories(categories)

    for category in selected:
        if progress_callback:
            progress_callback(f"Streaming {category} from Hugging Face...")
        stream = iter_remote_jsonl(huggingface_review_url(category, revision))
        count = 0
        for index, raw in enumerate(stream):
            if limit_per_category is not None and index >= limit_per_category:
                break
            yield normalize_review(dict(raw), category=category)
            count += 1
            if progress_callback and progress_every > 0 and count % progress_every == 0:
                progress_callback(f"  {category}: {count:,} reviews")
        if progress_callback:
            progress_callback(f"Finished {category}: {count:,} reviews")


# Stream metadata và chỉ giữ các parent_asin đang có trong catalog local.
def iter_huggingface_product_metadata(
    categories: Iterable[str],
    target_product_ids: Iterable[str],
    limit_per_category: int | None = None,
    revision: str = HF_REVISION,
    progress_callback: Callable[[str], None] | None = None,
    progress_every: int = 50_000,
) -> Iterator[dict]:
    if limit_per_category is not None and limit_per_category <= 0:
        raise ValueError("limit_per_category must be greater than zero or None.")
    selected = validate_metadata_categories(categories)
    remaining = {str(product_id) for product_id in target_product_ids if product_id}

    for category in selected:
        if not remaining:
            break
        if progress_callback:
            progress_callback(f"Streaming metadata {category}; còn {len(remaining):,} sản phẩm cần tìm...")
        stream = iter_remote_jsonl(huggingface_metadata_url(category, revision))
        scanned = 0
        matched = 0
        for raw in stream:
            if limit_per_category is not None and scanned >= limit_per_category:
                break
            scanned += 1
            product_id = str(raw.get("parent_asin") or "")
            if product_id in remaining:
                remaining.remove(product_id)
                matched += 1
                yield normalize_product_metadata(dict(raw), source_category=category)
            if progress_callback and progress_every > 0 and scanned % progress_every == 0:
                progress_callback(
                    f"  {category}: đã quét {scanned:,}, khớp {matched:,}, còn {len(remaining):,}"
                )
            if not remaining:
                break
        if progress_callback:
            progress_callback(
                f"Finished {category}: đã quét {scanned:,}, khớp {matched:,}, còn {len(remaining):,}"
            )


# Load một mẫu review có giới hạn để thử nghiệm local.
def load_review_sample(path: str | Path, limit: int = 10000) -> list[dict]:
    return list(iter_reviews(path, limit=limit))


# Load mẫu nhiều category từ Hugging Face; chủ yếu dùng cho notebook/test nhỏ.
def load_huggingface_review_sample(
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    limit_per_category: int = 100_000,
) -> list[dict]:
    return list(iter_huggingface_reviews(categories, limit_per_category))


# Trả về user id đầu tiên từ users.csv để demo gợi ý.
def first_user_id(users_path: str | Path) -> str:
    rows = read_csv_rows(users_path)
    if not rows:
        return ""
    return rows[0].get("user_id", "")
