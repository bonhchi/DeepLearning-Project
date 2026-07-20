# Tạo embedding text, image, metadata và embedding hợp nhất cho sản phẩm.

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from src.io_utils import json_loads_safe, parse_float, parse_int, write_csv_rows, write_jsonl
from src.vector_ops import concatenate_and_normalize, normalize, stable_hash_int, stable_random_vector


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "you",
    "are",
    "was",
    "were",
    "but",
    "not",
    "have",
    "very",
    "great",
    "good",
    "from",
    "they",
    "will",
    "just",
}


# Tokenize text sản phẩm thành các term chữ thường cho TF-IDF.
def tokenize(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


# Ghép các trường text của sản phẩm thành một document.
def product_text(product: dict) -> str:
    features = json_loads_safe(product.get("features"), default=[])
    if isinstance(features, list):
        features_text = " ".join(str(item) for item in features)
    else:
        features_text = str(features or "")
    return " ".join(
        [
            str(product.get("title", "")),
            str(product.get("category", "")),
            str(product.get("store", "")),
            str(product.get("description", "")),
            features_text,
        ]
    )


# Tạo vocabulary gọn, sắp xếp theo document frequency và tần suất từ.
def build_vocabulary(products: list[dict], max_features: int = 160, min_df: int = 1) -> list[str]:
    document_frequency: Counter[str] = Counter()
    total_frequency: Counter[str] = Counter()
    for product in products:
        tokens = tokenize(product_text(product))
        document_frequency.update(set(tokens))
        total_frequency.update(tokens)
    candidates = [
        token
        for token, df in document_frequency.items()
        if df >= min_df
    ]
    candidates.sort(key=lambda token: (document_frequency[token], total_frequency[token], token), reverse=True)
    return candidates[:max_features]


# Tính vector TF-IDF dense cho phần text của sản phẩm.
def build_text_embeddings(products: list[dict], vocabulary: list[str]) -> dict[str, list[float]]:
    vocab_index = {token: index for index, token in enumerate(vocabulary)}
    doc_tokens = {product["product_id"]: tokenize(product_text(product)) for product in products}
    document_count = max(len(products), 1)
    document_frequency = Counter()
    for tokens in doc_tokens.values():
        document_frequency.update(set(token for token in tokens if token in vocab_index))

    embeddings: dict[str, list[float]] = {}
    for product_id, tokens in doc_tokens.items():
        counts = Counter(token for token in tokens if token in vocab_index)
        vector = [0.0] * len(vocabulary)
        for token, count in counts.items():
            index = vocab_index[token]
            tf = 1.0 + math.log(count)
            idf = math.log((1 + document_count) / (1 + document_frequency[token])) + 1.0
            vector[index] = tf * idf
        embeddings[product_id] = normalize(vector)
    return embeddings


# Tạo image embedding ổn định từ URL ảnh hoặc product id.
def build_image_embedding(product: dict, dim: int = 32) -> list[float]:
    image_url = str(product.get("image_url", ""))
    key = image_url if image_url else f"missing_image::{product.get('product_id')}::{product.get('category')}"
    scale = 1.0 if image_url else 0.35
    return stable_random_vector(key, dim=dim, scale=scale)


# Mã hóa category, store, price, rating và popularity vào một vector.
def build_metadata_embedding(product: dict, dim: int = 32) -> list[float]:
    vector = [0.0] * dim
    price = parse_float(product.get("price"), 0.0)
    avg_rating = parse_float(product.get("average_rating"), 0.0)
    rating_number = parse_int(product.get("rating_number"), 0)
    vector[0] = min(price / 250.0, 1.0)
    vector[1] = min(avg_rating / 5.0, 1.0)
    vector[2] = min(math.log1p(rating_number) / 8.0, 1.0)
    vector[3] = 1.0 if product.get("preferred_price_range") == "budget" else 0.0
    vector[4 + stable_hash_int(str(product.get("category", "")), max(dim - 4, 1))] += 1.0
    vector[4 + stable_hash_int(str(product.get("store", "")), max(dim - 4, 1))] += 0.7
    return normalize(vector)


# Tạo embedding cho từng modality và vector hợp nhất cho mỗi sản phẩm.
def build_product_embeddings(
    products: list[dict],
    max_text_features: int = 160,
    image_dim: int = 32,
    metadata_dim: int = 32,
) -> tuple[list[dict], list[str]]:
    vocabulary = build_vocabulary(products, max_features=max_text_features)
    text_embeddings = build_text_embeddings(products, vocabulary)
    rows = []
    for product in products:
        product_id = product["product_id"]
        text_vector = text_embeddings.get(product_id, [0.0] * len(vocabulary))
        image_vector = build_image_embedding(product, dim=image_dim)
        metadata_vector = build_metadata_embedding(product, dim=metadata_dim)
        fused_vector = concatenate_and_normalize([metadata_vector, text_vector, image_vector])
        rows.append(
            {
                "product_id": product_id,
                "text_embedding": text_vector,
                "image_embedding": image_vector,
                "metadata_embedding": metadata_vector,
                "fused_embedding": fused_vector,
            }
        )
    return rows, vocabulary


# Tạo embedding sản phẩm và lưu ra JSONL cùng CSV.
def build_and_save_embeddings(
    products: list[dict],
    output_dir: str | Path,
    max_text_features: int = 160,
    image_dim: int = 32,
    metadata_dim: int = 32,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, vocabulary = build_product_embeddings(products, max_text_features, image_dim, metadata_dim)
    write_jsonl(output / "product_embeddings.jsonl", rows)
    write_csv_rows(
        output / "product_embeddings.csv",
        rows,
        ["product_id", "text_embedding", "image_embedding", "metadata_embedding", "fused_embedding"],
    )
    (output / "text_vocabulary.json").write_text(
        __import__("json").dumps(vocabulary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "product_embeddings": len(rows),
        "text_vocabulary_size": len(vocabulary),
        "fused_embedding_dim": len(rows[0]["fused_embedding"]) if rows else 0,
    }
