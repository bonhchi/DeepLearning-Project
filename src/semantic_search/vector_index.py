"""Cosine-similarity vector index with an optional FAISS backend.

The public API deliberately uses plain Python sequences and dictionaries.  That
keeps the index usable in the project's standard-library-only environment while
still taking advantage of FAISS (and NumPy) when they are installed.
"""

from __future__ import annotations

import json
import logging
import math
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.vector_ops import dot, normalize


LOGGER = logging.getLogger(__name__)
ARTIFACT_VERSION = 1


class VectorIndex:
    """Product vector index using inner product over L2-normalized vectors.

    Parameters
    ----------
    use_faiss:
        ``None`` (default) and ``True`` try FAISS first and transparently fall
        back to exact Python search when FAISS is unavailable.  ``False``
        always uses the deterministic exact backend.
    """

    def __init__(self, use_faiss: bool | None = None) -> None:
        self.use_faiss = use_faiss
        self.product_ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._dimension = 0
        self._backend = "unbuilt"
        self._faiss_index: Any | None = None
        self._vector_payload_path: Path | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return len(self.product_ids)

    @property
    def id_to_index(self) -> dict[str, int]:
        """Return a copy of the product-id to row mapping."""

        return {product_id: index for index, product_id in enumerate(self.product_ids)}

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_built(self) -> bool:
        return bool(self.product_ids) and self._dimension > 0

    def build(
        self,
        product_ids: Sequence[str] | Mapping[str, Sequence[float] | None],
        embeddings: Sequence[Sequence[float] | None] | None = None,
    ) -> "VectorIndex":
        """Build an index and validate catalog-to-vector alignment.

        The concise mapping form ``build({product_id: embedding})`` and the
        explicit parallel-list form ``build(product_ids, embeddings)`` are both
        supported.
        """

        ids, raw_vectors = self._coerce_input(product_ids, embeddings)
        vectors = self._validate_and_normalize(ids, raw_vectors)

        self.product_ids = ids
        self._vectors = vectors
        self._dimension = len(vectors[0])
        self._faiss_index = None
        self._vector_payload_path = None
        self._backend = "exact"
        self._try_build_faiss()
        return self

    def search(self, query_embedding: Sequence[float], top_k: int = 10) -> list[dict[str, Any]]:
        """Return the nearest products ordered by descending cosine score."""

        if not self.is_built:
            raise RuntimeError("Vector index has not been built or loaded")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query = self._coerce_vector(query_embedding, label="query embedding")
        if len(query) != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, got {len(query)}"
            )
        query = normalize(query)
        limit = min(top_k, self.size)

        if self._backend == "faiss" and self._faiss_index is not None:
            try:
                import numpy as np  # type: ignore

                scores, indices = self._faiss_index.search(
                    np.asarray([query], dtype="float32"), limit
                )
                return [
                    {
                        "product_id": self.product_ids[int(index)],
                        "score": float(score),
                        "index": int(index),
                    }
                    for score, index in zip(scores[0], indices[0])
                    if int(index) >= 0
                ]
            except Exception as exc:  # pragma: no cover - backend-specific failure
                LOGGER.warning("FAISS search failed; using exact fallback: %s", exc)

        self._ensure_exact_vectors()
        scored = [
            (dot(query, vector), index)
            for index, vector in enumerate(self._vectors)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "product_id": self.product_ids[index],
                "score": float(score),
                "index": index,
            }
            for score, index in scored[:limit]
        ]

    def save(self, path: str | Path) -> Path:
        """Save a self-contained artifact and return its metadata path.

        A ``.json`` target creates one portable file.  A directory target uses
        a compact float32 binary payload, which is preferable for real catalogs.
        """

        if not self.is_built:
            raise RuntimeError("Cannot save an index before build")

        target = Path(path)
        payload = {
            "artifact_version": ARTIFACT_VERSION,
            "product_ids": self.product_ids,
            "dimension": self.dimension,
            "size": self.size,
            "normalized": True,
            "backend_at_save": self.backend,
        }

        if target.suffix.lower() == ".json":
            target.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_exact_vectors()
            payload["vectors"] = self._vectors
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            temporary.replace(target)
            return target

        target.mkdir(parents=True, exist_ok=True)
        metadata_path = target / "metadata.json"
        vector_path = target / "vectors.f32"
        payload["vector_file"] = vector_path.name
        self._ensure_exact_vectors()
        flat = array("f", (value for vector in self._vectors for value in vector))
        temporary_vector = vector_path.with_suffix(vector_path.suffix + ".tmp")
        with temporary_vector.open("wb") as handle:
            flat.tofile(handle)
        temporary_vector.replace(vector_path)

        faiss_path = target / "index.faiss"
        if self.backend == "faiss" and self._faiss_index is not None:
            try:  # FAISS data is an optimization; vectors.f32 is authoritative.
                import faiss  # type: ignore

                temporary_faiss = target / "index.faiss.tmp"
                faiss.write_index(self._faiss_index, str(temporary_faiss))
                temporary_faiss.replace(faiss_path)
                payload["faiss_file"] = faiss_path.name
                payload["faiss_metric"] = "inner_product"
            except Exception as exc:  # pragma: no cover - optional dependency
                faiss_path.unlink(missing_ok=True)
                (target / "index.faiss.tmp").unlink(missing_ok=True)
                LOGGER.warning("Could not save optional FAISS artifact: %s", exc)
        else:
            # Never let a same-shape FAISS file from an older generation shadow
            # the newly written authoritative vectors.
            faiss_path.unlink(missing_ok=True)
        temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        temporary_metadata.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_metadata.replace(metadata_path)
        return metadata_path

    @classmethod
    def load(cls, path: str | Path, use_faiss: bool | None = None) -> "VectorIndex":
        """Load an artifact saved by :meth:`save` and rebuild its best backend."""

        target = Path(path)
        metadata_path = target / "metadata.json" if target.is_dir() else target
        if not metadata_path.exists():
            raise FileNotFoundError(f"Vector index artifact not found: {metadata_path}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError(
                "Unsupported vector index artifact version: "
                f"{payload.get('artifact_version')!r}"
            )

        ids = payload.get("product_ids")
        if not isinstance(ids, list):
            raise ValueError("Invalid vector index artifact: product_ids must be a list")
        dimension = payload.get("dimension")
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Invalid vector index artifact: dimension must be positive")

        vector_path: Path | None = None
        if "vectors" in payload:
            vectors = payload["vectors"]
        else:
            vector_name = payload.get("vector_file", "vectors.f32")
            vector_path = metadata_path.parent / str(vector_name)
            if not vector_path.exists():
                raise FileNotFoundError(f"Vector payload not found: {vector_path}")
            expected_bytes = len(ids) * dimension * array("f").itemsize
            if vector_path.stat().st_size != expected_bytes:
                raise ValueError(
                    "Corrupt vector payload: "
                    f"expected {expected_bytes} bytes, found {vector_path.stat().st_size}"
                )

            # A saved FAISS index can be loaded directly without materializing
            # millions of Python float objects. vectors.f32 remains the lazy,
            # authoritative fallback if FAISS search later fails.
            faiss_name = payload.get("faiss_file")
            faiss_path = metadata_path.parent / str(faiss_name or "index.faiss")
            if use_faiss is not False and faiss_name and faiss_path.exists():
                try:
                    import faiss  # type: ignore

                    faiss_index = faiss.read_index(str(faiss_path))
                    if int(faiss_index.d) != dimension or int(faiss_index.ntotal) != len(ids):
                        raise ValueError("FAISS artifact dimension/size mismatch")
                    if (
                        payload.get("faiss_metric") != "inner_product"
                        or int(faiss_index.metric_type) != int(faiss.METRIC_INNER_PRODUCT)
                    ):
                        raise ValueError("FAISS artifact metric mismatch")
                    index = cls(use_faiss=use_faiss)
                    index.product_ids = [str(value) for value in ids]
                    index._dimension = dimension
                    index._backend = "faiss"
                    index._faiss_index = faiss_index
                    index._vector_payload_path = vector_path
                    return index
                except (ImportError, ModuleNotFoundError):
                    pass
            vectors = cls._read_binary_vectors(vector_path, len(ids), dimension)

        index = cls(use_faiss=use_faiss)
        return index.build(ids, vectors)

    @staticmethod
    def _read_binary_vectors(
        vector_path: Path,
        size: int,
        dimension: int,
    ) -> list[list[float]]:
        flat = array("f")
        with vector_path.open("rb") as handle:
            flat.fromfile(handle, vector_path.stat().st_size // flat.itemsize)
        expected_values = size * dimension
        if len(flat) != expected_values:
            raise ValueError(
                "Corrupt vector payload: "
                f"expected {expected_values} values, found {len(flat)}"
            )
        return [
            list(flat[offset : offset + dimension])
            for offset in range(0, len(flat), dimension)
        ]

    def _ensure_exact_vectors(self) -> None:
        if self._vectors:
            return
        if self._vector_payload_path is None:
            raise RuntimeError("Exact vector payload is unavailable")
        self._vectors = self._read_binary_vectors(
            self._vector_payload_path,
            self.size,
            self.dimension,
        )

    @staticmethod
    def _coerce_input(
        product_ids: Sequence[str] | Mapping[str, Sequence[float] | None],
        embeddings: Sequence[Sequence[float] | None] | None,
    ) -> tuple[list[str], list[Sequence[float] | None]]:
        if isinstance(product_ids, Mapping):
            if embeddings is not None:
                raise ValueError("embeddings must be omitted when product_ids is a mapping")
            return [str(key) for key in product_ids], list(product_ids.values())
        if isinstance(product_ids, (str, bytes)):
            raise TypeError("product_ids must be a sequence of ids, not a string")
        if embeddings is None:
            raise ValueError("embeddings are required when product_ids is a sequence")
        return [str(product_id) for product_id in product_ids], list(embeddings)

    @classmethod
    def _validate_and_normalize(
        cls,
        product_ids: list[str],
        raw_vectors: list[Sequence[float] | None],
    ) -> list[list[float]]:
        if not product_ids:
            raise ValueError("Cannot build an index without products")
        if len(product_ids) != len(raw_vectors):
            raise ValueError(
                "Catalog/embedding count mismatch: "
                f"{len(product_ids)} product ids and {len(raw_vectors)} embeddings"
            )

        missing_ids = [index for index, product_id in enumerate(product_ids) if not product_id.strip()]
        if missing_ids:
            raise ValueError(f"Missing product_id at positions: {missing_ids[:5]}")
        seen: set[str] = set()
        duplicate_set: set[str] = set()
        for product_id in product_ids:
            if product_id in seen:
                duplicate_set.add(product_id)
            seen.add(product_id)
        duplicates = sorted(duplicate_set)
        if duplicates:
            raise ValueError(f"Duplicate product_id values: {duplicates[:5]}")

        missing_embeddings = [
            product_ids[index]
            for index, vector in enumerate(raw_vectors)
            if vector is None
        ]
        if missing_embeddings:
            raise ValueError(f"Missing embeddings for product ids: {missing_embeddings[:5]}")

        vectors = [
            cls._coerce_vector(vector, label=f"embedding for {product_ids[index]!r}")
            for index, vector in enumerate(raw_vectors)
            if vector is not None
        ]
        dimension = len(vectors[0])
        if dimension == 0:
            raise ValueError("Embedding dimension must be greater than zero")
        mismatches = [
            f"{product_ids[index]}:{len(vector)}"
            for index, vector in enumerate(vectors)
            if len(vector) != dimension
        ]
        if mismatches:
            raise ValueError(
                f"Embedding dimension mismatch; expected {dimension}, got {mismatches[:5]}"
            )
        return [normalize(vector) for vector in vectors]

    @staticmethod
    def _coerce_vector(vector: Sequence[float], label: str) -> list[float]:
        if isinstance(vector, (str, bytes)):
            raise TypeError(f"{label} must be a numeric sequence")
        try:
            output = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label} must contain only numeric values") from exc
        if not output:
            raise ValueError(f"{label} must not be empty")
        if not all(math.isfinite(value) for value in output):
            raise ValueError(f"{label} contains NaN or infinite values")
        return output

    def _try_build_faiss(self) -> None:
        if self.use_faiss is False:
            return
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore

            faiss_index = faiss.IndexFlatIP(self.dimension)
            faiss_index.add(np.asarray(self._vectors, dtype="float32"))
            self._faiss_index = faiss_index
            self._backend = "faiss"
        except (ImportError, ModuleNotFoundError):
            LOGGER.info("FAISS is unavailable; using exact cosine search")
        except Exception as exc:  # pragma: no cover - optional backend edge case
            LOGGER.warning("Could not initialize FAISS; using exact fallback: %s", exc)


# Descriptive alias for callers that prefer the domain-specific class name.
SemanticVectorIndex = VectorIndex


__all__ = ["SemanticVectorIndex", "VectorIndex"]
