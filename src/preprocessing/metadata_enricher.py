# Ghép metadata catalog Amazon vào products.csv để bổ sung ảnh và tên sản phẩm thật.

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.data.amazon_reviews import METADATA_CATEGORIES, iter_huggingface_product_metadata
from src.io_utils import parse_float, parse_int, read_csv_rows, write_csv_rows
from src.preprocessing.dataset_builder import (
    AMAZON_ITEM_METADATA_SOURCE,
    PRODUCT_FIELDS,
    parse_metadata_provenance,
    price_bucket,
)


def infer_metadata_categories(products: list[dict]) -> list[str]:
    """Suy ra file metadata từ category; catalog Fashion cũ dùng một file chung."""
    source_categories = {
        str(product.get("category", ""))
        for product in products
        if product.get("category") in METADATA_CATEGORIES
    }
    fashion_categories = {
        "tops", "dresses", "shoes", "outerwear", "bottoms", "socks", "bags",
        "watches", "eyewear", "jewelry", "fashion_accessories", "Fashion",
    }
    if any(str(product.get("category", "")) in fashion_categories for product in products):
        source_categories.add("Amazon_Fashion")
    return sorted(source_categories) if source_categories else ["Amazon_Fashion"]


def apply_product_metadata(product: dict, metadata: dict) -> bool:
    """Cập nhật một product và trả về True nếu thêm được URL ảnh catalog."""
    had_image = bool(str(product.get("image_url", "")).strip())
    updated_fields: list[str] = []
    image_url = str(metadata.get("image_url", "")).strip()
    if image_url:
        product["image_url"] = image_url
        updated_fields.append("image_url")
    if metadata.get("title"):
        product["title"] = metadata["title"]
        updated_fields.append("title")
    if metadata.get("store"):
        product["store"] = metadata["store"]
        updated_fields.append("store")
    if metadata.get("description"):
        product["description"] = metadata["description"]
        updated_fields.append("description")
    if metadata.get("features"):
        product["features"] = metadata["features"]
        updated_fields.append("features")
    price = parse_float(metadata.get("price"), 0.0)
    if price > 0:
        product["price"] = price
        product["preferred_price_range"] = price_bucket(price)
        updated_fields.extend(["price", "preferred_price_range"])
    average_rating = parse_float(metadata.get("average_rating"), 0.0)
    if average_rating > 0:
        product["average_rating"] = average_rating
        updated_fields.append("average_rating")
    rating_number = parse_int(metadata.get("rating_number"), 0)
    if rating_number > 0:
        product["rating_number"] = rating_number
        updated_fields.append("rating_number")
    if updated_fields:
        provenance = parse_metadata_provenance(product)
        provenance.update({field: AMAZON_ITEM_METADATA_SOURCE for field in updated_fields})
        product["metadata_source"] = AMAZON_ITEM_METADATA_SOURCE
        product["metadata_provenance"] = provenance
    return not had_image and bool(image_url)


def enrich_product_images(
    products_path: str | Path,
    categories: Iterable[str] | None = None,
    limit_per_category: int | None = None,
    progress_callback=print,
) -> dict:
    """Stream metadata cần thiết, cập nhật products.csv và báo cáo độ phủ ảnh."""
    products = read_csv_rows(products_path)
    selected_categories = list(categories or infer_metadata_categories(products))
    product_map = {product["product_id"]: product for product in products}
    missing_before = {
        product_id
        for product_id, product in product_map.items()
        if not str(product.get("image_url", "")).strip()
    }
    matched_metadata = 0
    images_added = 0
    for metadata in iter_huggingface_product_metadata(
        selected_categories,
        product_map,
        limit_per_category=limit_per_category,
        progress_callback=progress_callback,
    ):
        product = product_map.get(metadata["product_id"])
        if product is None:
            continue
        matched_metadata += 1
        images_added += int(apply_product_metadata(product, metadata))

    write_csv_rows(products_path, products, PRODUCT_FIELDS)
    missing_after = sum(not str(product.get("image_url", "")).strip() for product in products)
    return {
        "categories": selected_categories,
        "products": len(products),
        "missing_images_before": len(missing_before),
        "metadata_matched": matched_metadata,
        "images_added": images_added,
        "missing_images_after": missing_after,
        "image_coverage_percent": round(
            100.0 * (len(products) - missing_after) / max(len(products), 1), 2
        ),
    }
