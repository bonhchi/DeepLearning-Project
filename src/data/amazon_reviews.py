# Bộ đọc dữ liệu Amazon Reviews 2023 từ JSONL local hoặc Hugging Face.

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

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


# Tạo URL resolve trực tiếp tới JSONL trên Hugging Face Hub.
def huggingface_review_url(category: str, revision: str = HF_REVISION) -> str:
    return (
        f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/{revision}/"
        f"raw/review_categories/{category}.jsonl"
    )


# Kiểm tra và loại category trùng nhưng vẫn giữ nguyên thứ tự đầu vào.
def validate_categories(categories: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(category).strip() for category in categories))
    if not selected:
        raise ValueError("At least one Amazon Reviews category is required.")
    invalid = [category for category in selected if category not in DEFAULT_CATEGORIES]
    if invalid:
        allowed = ", ".join(DEFAULT_CATEGORIES)
        raise ValueError(f"Unsupported categories: {', '.join(invalid)}. Allowed: {allowed}")
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
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face mode requires the 'datasets' package. "
            "Install it with: pip install -r requirements.txt"
        ) from exc

    for category in selected:
        if progress_callback:
            progress_callback(f"Streaming {category} from Hugging Face...")
        stream = load_dataset(
            "json",
            data_files={"train": huggingface_review_url(category, revision)},
            split="train",
            streaming=True,
        )
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
