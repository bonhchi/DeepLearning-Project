"""Memory-efficient sparse exact index for normalized TF-IDF vectors."""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.feature_extraction.embeddings import TfidfEncoder, lexical_tokenize


ARTIFACT_VERSION = 1


class SparseTfidfIndex:
    """Inverted index implementing the subset of ``VectorIndex`` used by search."""

    def __init__(self) -> None:
        self.product_ids: list[str] = []
        self._dimension = 0
        self._postings: dict[int, list[tuple[int, float]]] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return len(self.product_ids)

    @property
    def backend(self) -> str:
        return "sparse_exact"

    @property
    def is_built(self) -> bool:
        return bool(self.product_ids) and self.dimension > 0

    def build(
        self,
        product_ids: Sequence[str],
        documents: Sequence[str],
        encoder: TfidfEncoder,
    ) -> "SparseTfidfIndex":
        ids = [str(product_id) for product_id in product_ids]
        if not ids:
            raise ValueError("Cannot build a lexical index without products")
        if len(ids) != len(documents):
            raise ValueError("Catalog/document count mismatch for lexical index")
        if any(not product_id.strip() for product_id in ids):
            raise ValueError("Lexical index contains an empty product_id")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate product_id values in lexical index")
        if not encoder.is_fitted or encoder.dimension <= 0:
            raise ValueError("TfidfEncoder must be fitted before indexing")

        postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
        vocab_index = {token: index for index, token in enumerate(encoder.vocabulary)}
        for document_index, document in enumerate(documents):
            counts = Counter(
                token for token in lexical_tokenize(str(document)) if token in vocab_index
            )
            weighted = {
                vocab_index[token]: (1.0 + math.log(count)) * encoder.idf[token]
                for token, count in counts.items()
            }
            norm = math.sqrt(sum(value * value for value in weighted.values()))
            scale = 1.0 / norm if encoder.normalize_embeddings and norm else 1.0
            for token_index, value in weighted.items():
                postings[token_index].append((document_index, value * scale))

        self.product_ids = ids
        self._dimension = encoder.dimension
        self._postings = dict(postings)
        return self

    def search(self, query_embedding: Sequence[float], top_k: int = 10) -> list[dict[str, Any]]:
        if not self.is_built:
            raise RuntimeError("Sparse TF-IDF index has not been built or loaded")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query = [float(value) for value in query_embedding]
        if len(query) != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, got {len(query)}"
            )
        if any(not math.isfinite(value) for value in query):
            raise ValueError("Query embedding contains NaN or infinite values")

        scores: dict[int, float] = defaultdict(float)
        for token_index, query_value in enumerate(query):
            if not query_value:
                continue
            for document_index, document_value in self._postings.get(token_index, ()):
                scores[document_index] += query_value * document_value
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        limit = min(top_k, self.size)
        return [
            {
                "product_id": self.product_ids[index],
                "score": float(score),
                "index": index,
            }
            for index, score in ranked[:limit]
        ]

    def save(self, path: str | Path) -> Path:
        if not self.is_built:
            raise RuntimeError("Cannot save a lexical index before build")
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        postings_path = target / "postings.json.gz"
        metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "artifact_type": "sparse_tfidf",
            "product_ids": self.product_ids,
            "dimension": self.dimension,
            "size": self.size,
            "postings_file": postings_path.name,
        }
        temporary_postings = postings_path.with_suffix(postings_path.suffix + ".tmp")
        with gzip.open(temporary_postings, "wt", encoding="utf-8") as handle:
            json.dump(
                [
                    [token_index, postings]
                    for token_index, postings in sorted(self._postings.items())
                ],
                handle,
                separators=(",", ":"),
            )
        temporary_postings.replace(postings_path)
        metadata_path = target / "metadata.json"
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary_metadata.replace(metadata_path)
        return metadata_path

    @classmethod
    def load(cls, path: str | Path) -> "SparseTfidfIndex":
        target = Path(path)
        metadata_path = target / "metadata.json" if target.is_dir() else target
        if not metadata_path.exists():
            raise FileNotFoundError(f"Sparse TF-IDF artifact not found: {metadata_path}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("Unsupported sparse TF-IDF artifact version")
        if payload.get("artifact_type") != "sparse_tfidf":
            raise ValueError("Artifact is not a sparse TF-IDF index")
        product_ids = [str(value) for value in payload.get("product_ids", [])]
        dimension = int(payload.get("dimension", 0))
        if not product_ids or dimension <= 0 or len(set(product_ids)) != len(product_ids):
            raise ValueError("Corrupt sparse TF-IDF metadata")
        postings_path = metadata_path.parent / str(
            payload.get("postings_file", "postings.json.gz")
        )
        if not postings_path.exists():
            raise FileNotFoundError(f"Sparse TF-IDF postings not found: {postings_path}")
        with gzip.open(postings_path, "rt", encoding="utf-8") as handle:
            raw_postings = json.load(handle)
        index = cls()
        index.product_ids = product_ids
        index._dimension = dimension
        index._postings = {
            int(token_index): [
                (int(document_index), float(value))
                for document_index, value in postings
            ]
            for token_index, postings in raw_postings
        }
        if any(
            token_index < 0
            or token_index >= dimension
            or any(doc_index < 0 or doc_index >= len(product_ids) for doc_index, _ in postings)
            for token_index, postings in index._postings.items()
        ):
            raise ValueError("Corrupt sparse TF-IDF postings")
        return index


def load_lexical_index(path: str | Path) -> SparseTfidfIndex:
    return SparseTfidfIndex.load(path)


__all__ = ["SparseTfidfIndex", "load_lexical_index"]
