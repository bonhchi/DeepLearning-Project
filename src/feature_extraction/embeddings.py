# Tạo embedding text, image, metadata và embedding hợp nhất cho sản phẩm.

from __future__ import annotations

import json
import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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


LOGGER = logging.getLogger(__name__)
UNICODE_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def lexical_tokenize(text: str) -> list[str]:
    """Tokenize Unicode text for the reusable TF-IDF encoder.

    Legacy product-embedding helpers continue to use :func:`tokenize`, keeping
    their artifact format and dimensions unchanged.
    """

    tokens = UNICODE_TOKEN_PATTERN.findall(str(text).casefold())
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def _artifact_path(path: str | Path, default_name: str, create: bool = False) -> Path:
    """Resolve either a JSON file target or an encoder artifact directory."""

    target = Path(path)
    if target.suffix.lower() == ".json":
        if create:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target / default_name


class TextEncoder(ABC):
    """Common interface implemented by lexical and dense text encoders."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension, or zero before an encoder is fitted."""

    def fit(self, texts: Sequence[str]) -> "TextEncoder":
        """Fit stateful encoders; stateless dense encoders need no training."""

        return self

    @abstractmethod
    def encode(
        self,
        texts: str | Sequence[str],
        batch_size: int = 32,
        use_cache: bool = True,
    ) -> list[float] | list[list[float]]:
        """Encode one string or a batch while preserving the input shape."""

    def encode_one(self, text: str, use_cache: bool = True) -> list[float]:
        vector = self.encode(text, use_cache=use_cache)
        return list(vector)  # type: ignore[arg-type]

    @abstractmethod
    def save(self, path: str | Path, include_cache: bool = True) -> Path:
        """Persist encoder configuration and fitted state."""


class TfidfEncoder(TextEncoder):
    """Small dependency-free TF-IDF encoder used as the lexical baseline."""

    ARTIFACT_VERSION = 1

    def __init__(
        self,
        max_features: int = 160,
        min_df: int = 1,
        normalize_embeddings: bool = True,
        cache_enabled: bool = True,
    ) -> None:
        if max_features <= 0:
            raise ValueError("max_features must be greater than zero")
        if min_df <= 0:
            raise ValueError("min_df must be greater than zero")
        self.max_features = max_features
        self.min_df = min_df
        self.normalize_embeddings = normalize_embeddings
        self.cache_enabled = cache_enabled
        self.vocabulary: list[str] = []
        self.idf: dict[str, float] = {}
        self._vocab_index: dict[str, int] = {}
        self._cache: dict[str, list[float]] = {}
        self._is_fitted = False

    @property
    def dimension(self) -> int:
        return len(self.vocabulary)

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, texts: Sequence[str]) -> "TfidfEncoder":
        if isinstance(texts, (str, bytes)):
            raise TypeError("fit expects a sequence of documents, not one string")
        documents = [str(text) for text in texts]
        document_frequency: Counter[str] = Counter()
        total_frequency: Counter[str] = Counter()
        tokenized: list[list[str]] = []
        for text in documents:
            tokens = lexical_tokenize(text)
            tokenized.append(tokens)
            document_frequency.update(set(tokens))
            total_frequency.update(tokens)

        candidates = [
            token
            for token, frequency in document_frequency.items()
            if frequency >= self.min_df
        ]
        candidates.sort(
            key=lambda token: (
                document_frequency[token],
                total_frequency[token],
                token,
            ),
            reverse=True,
        )
        self.vocabulary = candidates[: self.max_features]
        if not self.vocabulary:
            # A one-dimensional zero-vector space keeps tiny/empty-token test
            # corpora searchable without changing the legacy embedding API.
            self.vocabulary = ["__empty__"]
        self._vocab_index = {
            token: index for index, token in enumerate(self.vocabulary)
        }
        document_count = max(len(documents), 1)
        self.idf = {
            token: math.log(
                (1 + document_count) / (1 + document_frequency[token])
            )
            + 1.0
            for token in self.vocabulary
        }
        self._cache.clear()
        self._is_fitted = True
        return self

    def encode(
        self,
        texts: str | Sequence[str],
        batch_size: int = 32,
        use_cache: bool = True,
    ) -> list[float] | list[list[float]]:
        del batch_size  # Kept for interface parity; TF-IDF is encoded per document.
        if not self.is_fitted:
            raise RuntimeError("TfidfEncoder must be fitted before encode")
        single = isinstance(texts, str)
        documents = [texts] if single else [str(text) for text in texts]
        vectors: list[list[float]] = []
        for text in documents:
            cached = self._cache.get(text) if use_cache and self.cache_enabled else None
            if cached is not None:
                vectors.append(list(cached))
                continue
            counts = Counter(
                token for token in lexical_tokenize(text) if token in self._vocab_index
            )
            vector = [0.0] * self.dimension
            for token, count in counts.items():
                vector[self._vocab_index[token]] = (1.0 + math.log(count)) * self.idf[token]
            if self.normalize_embeddings:
                vector = normalize(vector)
            if use_cache and self.cache_enabled:
                self._cache[text] = list(vector)
            vectors.append(vector)
        return vectors[0] if single else vectors

    def save(self, path: str | Path, include_cache: bool = True) -> Path:
        if not self.is_fitted:
            raise RuntimeError("Cannot save TfidfEncoder before fit")
        artifact = _artifact_path(path, "tfidf_encoder.json", create=True)
        payload = {
            "artifact_version": self.ARTIFACT_VERSION,
            "encoder_type": "tfidf",
            "max_features": self.max_features,
            "min_df": self.min_df,
            "normalize_embeddings": self.normalize_embeddings,
            "cache_enabled": self.cache_enabled,
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "cache": self._cache if include_cache else {},
        }
        artifact.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        return artifact

    @classmethod
    def load(cls, path: str | Path) -> "TfidfEncoder":
        artifact = _artifact_path(path, "tfidf_encoder.json")
        if not artifact.exists():
            raise FileNotFoundError(f"TF-IDF artifact not found: {artifact}")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if payload.get("artifact_version") != cls.ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported TF-IDF artifact version: {payload.get('artifact_version')!r}"
            )
        encoder = cls(
            max_features=int(payload["max_features"]),
            min_df=int(payload["min_df"]),
            normalize_embeddings=bool(payload.get("normalize_embeddings", True)),
            cache_enabled=bool(payload.get("cache_enabled", True)),
        )
        encoder.vocabulary = [str(token) for token in payload.get("vocabulary", [])]
        encoder._vocab_index = {
            token: index for index, token in enumerate(encoder.vocabulary)
        }
        encoder.idf = {
            str(token): float(value) for token, value in payload.get("idf", {}).items()
        }
        if set(encoder.vocabulary) != set(encoder.idf):
            raise ValueError("Corrupt TF-IDF artifact: vocabulary and idf do not align")
        encoder._cache = {
            str(text): [float(value) for value in vector]
            for text, vector in payload.get("cache", {}).items()
        }
        if any(len(vector) != encoder.dimension for vector in encoder._cache.values()):
            raise ValueError("Corrupt TF-IDF artifact: cached vector dimension mismatch")
        encoder._is_fitted = True
        return encoder


class DenseTextEncoder(TextEncoder):
    """Multilingual Sentence Transformer with a deterministic offline fallback.

    ``backend='auto'`` first attempts to load the configured model through
    ``sentence-transformers``.  The default ``local_files_only=True`` prevents
    accidental model downloads in tests and offline jobs.  If the package or
    local model is unavailable, a normalized feature-hashing encoder is used.
    """

    ARTIFACT_VERSION = 2
    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    VERIFICATION_TEXT = "intent search encoder artifact verification"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dimension: int = 384,
        normalize_embeddings: bool = True,
        cache_enabled: bool = True,
        backend: str = "auto",
        local_files_only: bool = True,
        model: Any | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        if backend not in {"auto", "sentence-transformers", "fallback"}:
            raise ValueError("backend must be auto, sentence-transformers, or fallback")
        self.model_name = model_name
        self.fallback_dimension = dimension
        self.normalize_embeddings = normalize_embeddings
        self.cache_enabled = cache_enabled
        self.requested_backend = backend
        self.local_files_only = local_files_only
        self._model = model
        self._backend_name = "sentence-transformers" if model is not None else "uninitialized"
        self._observed_dimension = 0
        self._cache: dict[str, list[float]] = {}
        self._expected_verification_vector: list[float] | None = None
        self._verification_checked = False

    @property
    def dimension(self) -> int:
        return self._observed_dimension or self.fallback_dimension

    @property
    def backend_name(self) -> str:
        if self._backend_name == "uninitialized":
            self._ensure_backend()
        return self._backend_name

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def encode(
        self,
        texts: str | Sequence[str],
        batch_size: int = 32,
        use_cache: bool = True,
    ) -> list[float] | list[list[float]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        single = isinstance(texts, str)
        documents = [texts] if single else [str(text) for text in texts]
        if not documents:
            return []

        self._ensure_backend()
        missing = list(
            dict.fromkeys(
                text
                for text in documents
                if not (use_cache and self.cache_enabled and text in self._cache)
            )
        )
        generated: dict[str, list[float]] = {}
        if missing:
            if self._backend_name == "sentence-transformers":
                try:
                    generated = self._encode_sentence_transformer(missing, batch_size)
                except Exception as exc:
                    if self.requested_backend == "sentence-transformers":
                        raise RuntimeError("Sentence Transformer encoding failed") from exc
                    LOGGER.warning(
                        "Sentence Transformer encoding failed; using deterministic fallback: %s",
                        exc,
                    )
                    self._model = None
                    self._backend_name = "fallback"
                    generated = {text: self._fallback_vector(text) for text in missing}
            else:
                generated = {text: self._fallback_vector(text) for text in missing}

            for text, vector in generated.items():
                if self._observed_dimension and len(vector) != self._observed_dimension:
                    raise ValueError(
                        "Dense encoder dimension changed from "
                        f"{self._observed_dimension} to {len(vector)}"
                    )
                self._observed_dimension = len(vector)
                if use_cache and self.cache_enabled:
                    self._cache[text] = list(vector)

        vectors = [
            list(self._cache[text])
            if use_cache and self.cache_enabled and text in self._cache
            else list(generated[text])
            for text in documents
        ]
        return vectors[0] if single else vectors

    def save(self, path: str | Path, include_cache: bool = True) -> Path:
        artifact = _artifact_path(path, "dense_text_encoder.json", create=True)
        self._ensure_backend()
        verification_vector = self._backend_vector(self.VERIFICATION_TEXT)
        payload = {
            "artifact_version": self.ARTIFACT_VERSION,
            "encoder_type": "dense_text",
            "model_name": self.model_name,
            "dimension": self.dimension,
            "normalize_embeddings": self.normalize_embeddings,
            "cache_enabled": self.cache_enabled,
            "requested_backend": self.requested_backend,
            "backend_at_save": self.backend_name,
            "local_files_only": self.local_files_only,
            "verification_text": self.VERIFICATION_TEXT,
            "verification_vector": [round(value, 8) for value in verification_vector],
            "cache": self._cache if include_cache else {},
        }
        temporary = artifact.with_suffix(artifact.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(artifact)
        return artifact

    @classmethod
    def load(cls, path: str | Path) -> "DenseTextEncoder":
        artifact = _artifact_path(path, "dense_text_encoder.json")
        if not artifact.exists():
            raise FileNotFoundError(f"Dense encoder artifact not found: {artifact}")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if payload.get("artifact_version") != cls.ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported dense artifact version: {payload.get('artifact_version')!r}"
            )
        dimension = int(payload.get("dimension", 384))
        backend_at_save = str(payload.get("backend_at_save", "fallback"))
        # The vectors in a semantic index and its query encoder must come from
        # the same embedding space.  In particular, an artifact created by the
        # deterministic fallback must not silently switch to a newly installed
        # Sentence Transformer after reload.
        if backend_at_save == "fallback":
            load_backend = "fallback"
        elif backend_at_save == "sentence-transformers":
            load_backend = "sentence-transformers"
        else:
            raise ValueError(
                f"Unsupported dense backend in artifact: {backend_at_save!r}"
            )
        encoder = cls(
            model_name=str(payload.get("model_name", cls.DEFAULT_MODEL)),
            dimension=dimension,
            normalize_embeddings=bool(payload.get("normalize_embeddings", True)),
            cache_enabled=bool(payload.get("cache_enabled", True)),
            backend=load_backend,
            local_files_only=bool(payload.get("local_files_only", True)),
        )
        encoder._cache = {
            str(text): [float(value) for value in vector]
            for text, vector in payload.get("cache", {}).items()
        }
        if any(len(vector) != dimension for vector in encoder._cache.values()):
            raise ValueError("Corrupt dense artifact: cached vector dimension mismatch")
        encoder._observed_dimension = dimension
        expected_verification = payload.get("verification_vector")
        if not isinstance(expected_verification, list) or len(expected_verification) != dimension:
            raise ValueError("Corrupt dense artifact: missing encoder verification vector")
        encoder._expected_verification_vector = [
            float(value) for value in expected_verification
        ]
        return encoder

    def _ensure_backend(self) -> None:
        if self._backend_name != "uninitialized":
            return
        if self.requested_backend == "fallback":
            self._backend_name = "fallback"
            self._verify_loaded_backend()
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=self.local_files_only,
                )
            except TypeError as exc:
                if self.local_files_only:
                    raise RuntimeError(
                        "Installed sentence-transformers does not support safe local-only loading"
                    ) from exc
                self._model = SentenceTransformer(self.model_name)
            self._backend_name = "sentence-transformers"
            self._verify_loaded_backend()
        except Exception as exc:
            if self.requested_backend == "sentence-transformers":
                raise RuntimeError(
                    f"Cannot load Sentence Transformer model {self.model_name!r}"
                ) from exc
            LOGGER.info("Using deterministic dense fallback: %s", exc)
            self._model = None
            self._backend_name = "fallback"
            self._verify_loaded_backend()

    def _backend_vector(self, text: str) -> list[float]:
        if self._backend_name == "sentence-transformers":
            return self._encode_sentence_transformer([text], 1)[text]
        return self._fallback_vector(text)

    def _verify_loaded_backend(self) -> None:
        if self._verification_checked or self._expected_verification_vector is None:
            return
        actual = self._backend_vector(self.VERIFICATION_TEXT)
        expected = self._expected_verification_vector
        if len(actual) != len(expected) or max(
            (abs(left - right) for left, right in zip(actual, expected)),
            default=float("inf"),
        ) > 1e-5:
            raise RuntimeError(
                "Dense encoder model snapshot differs from the one used to build the index"
            )
        self._verification_checked = True

    def _encode_sentence_transformer(
        self,
        texts: list[str],
        batch_size: int,
    ) -> dict[str, list[float]]:
        encoded = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        vectors = [[float(value) for value in vector] for vector in encoded]
        if self.normalize_embeddings:
            vectors = [normalize(vector) for vector in vectors]
        return dict(zip(texts, vectors))

    def _fallback_vector(self, text: str) -> list[float]:
        """Feature hashing over multilingual words, bigrams, and character trigrams."""

        normalized_text = " ".join(UNICODE_TOKEN_PATTERN.findall(text.casefold()))
        words = [word for word in normalized_text.split() if len(word) > 1]
        features: list[str] = [f"w:{word}" for word in words]
        features.extend(
            f"b:{left}_{right}" for left, right in zip(words, words[1:])
        )
        compact = normalized_text.replace(" ", "_")
        features.extend(
            f"c:{compact[index:index + 3]}"
            for index in range(max(0, len(compact) - 2))
        )

        vector = [0.0] * self.fallback_dimension
        for feature in features:
            bucket = stable_hash_int(feature, self.fallback_dimension)
            sign = 1.0 if stable_hash_int(f"sign::{feature}", 2) == 0 else -1.0
            vector[bucket] += sign
        return normalize(vector) if self.normalize_embeddings else vector


# Backwards-friendly descriptive alias for callers that prefer "Base" naming.
BaseTextEncoder = TextEncoder


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
