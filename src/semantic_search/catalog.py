"""Catalog enrichment and fingerprint helpers for search serving artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.feature_extraction.embeddings import product_text
from src.io_utils import parse_float


FINGERPRINT_VERSION = 1


def attach_business_context(
    products: Iterable[Mapping[str, Any]],
    business_rows: Iterable[Mapping[str, Any]],
    *,
    availability_threshold: float = 0.30,
) -> list[dict[str, Any]]:
    """Attach inventory signals and a deterministic availability flag.

    ``business_context.csv`` is the project's ranking/demo inventory source.  If
    no row exists for a product, an explicit availability field already present
    in the catalog is retained; otherwise availability remains unknown.
    """

    if not 0.0 <= availability_threshold <= 1.0:
        raise ValueError("availability_threshold must be between 0 and 1")
    context_by_id = {
        str(row.get("product_id", "")): dict(row)
        for row in business_rows
        if str(row.get("product_id", "")).strip()
    }
    enriched: list[dict[str, Any]] = []
    for raw_product in products:
        product = dict(raw_product)
        product_id = str(product.get("product_id", "")).strip()
        context = context_by_id.get(product_id)
        if context is not None:
            inventory = parse_float(context.get("inventory_score"), 0.0)
            product.update(
                {
                    "inventory_score": inventory,
                    "margin_score": parse_float(context.get("margin_score"), 0.0),
                    "discount_rate": parse_float(context.get("discount_rate"), 0.0),
                    "risk_score": parse_float(context.get("risk_score"), 0.0),
                    "campaign_eligible": context.get("campaign_eligible", ""),
                    "available": inventory >= availability_threshold,
                    "availability_source": "business_context_inventory_proxy",
                }
            )
        enriched.append(product)
    return enriched


def search_catalog_fingerprint(products: Sequence[Mapping[str, Any]]) -> str:
    """Hash ordered retrieval inputs so stale indices fail closed at runtime."""

    digest = hashlib.sha256()
    digest.update(f"search-catalog-v{FINGERPRINT_VERSION}\n".encode())
    for product in products:
        payload = {
            "product_id": str(product.get("product_id", "")),
            "text": product_text(dict(product)),
            "available": str(product.get("available", "")),
            "inventory_score": str(product.get("inventory_score", "")),
        }
        digest.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def artifact_fingerprint(path: str | Path) -> str:
    """Hash one artifact file or all files below an artifact directory."""

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Search artifact not found: {target}")
    files = [target] if target.is_file() else sorted(
        file_path for file_path in target.rglob("*") if file_path.is_file()
    )
    digest = hashlib.sha256()
    for file_path in files:
        relative = file_path.name if target.is_file() else file_path.relative_to(target).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def write_search_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "artifact_version": FINGERPRINT_VERSION,
        **dict(payload),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(target)
    return target


def validate_search_manifest(
    path: str | Path,
    products: Sequence[Mapping[str, Any]],
    *,
    expected_product_ids: Sequence[str],
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"Search index manifest not found: {target}. Re-run index-semantic."
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("artifact_version") != FINGERPRINT_VERSION:
        raise ValueError("Unsupported search index manifest version")
    manifest_ids = [str(value) for value in payload.get("product_ids", [])]
    actual_ids = [str(value) for value in expected_product_ids]
    if manifest_ids != actual_ids:
        raise ValueError("Search index manifest product IDs do not match the loaded indices")
    actual_fingerprint = search_catalog_fingerprint(products)
    if payload.get("catalog_fingerprint") != actual_fingerprint:
        raise ValueError(
            "Search catalog changed after indexing; re-run index-semantic before serving"
        )
    expected_artifacts = payload.get("artifact_fingerprints", {})
    for name, artifact_path in (artifact_paths or {}).items():
        if expected_artifacts.get(name) != artifact_fingerprint(artifact_path):
            raise ValueError(
                f"Search artifact {name!r} changed after indexing; re-run index-semantic"
            )
    return payload


__all__ = [
    "attach_business_context",
    "artifact_fingerprint",
    "search_catalog_fingerprint",
    "validate_search_manifest",
    "write_search_manifest",
]
