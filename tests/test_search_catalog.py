import tempfile
import unittest
from pathlib import Path

from src.semantic_search.catalog import (
    artifact_fingerprint,
    attach_business_context,
    search_catalog_fingerprint,
    validate_search_manifest,
    write_search_manifest,
)


class SearchCatalogTests(unittest.TestCase):
    def test_business_inventory_proxy_sets_availability(self):
        products = [{"product_id": "low"}, {"product_id": "high"}, {"product_id": "unknown"}]
        rows = [
            {"product_id": "low", "inventory_score": "0.1"},
            {"product_id": "high", "inventory_score": "0.8"},
        ]
        enriched = attach_business_context(products, rows, availability_threshold=0.3)
        by_id = {row["product_id"]: row for row in enriched}
        self.assertFalse(by_id["low"]["available"])
        self.assertTrue(by_id["high"]["available"])
        self.assertNotIn("available", by_id["unknown"])

    def test_manifest_rejects_catalog_or_artifact_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "encoder.json"
            artifact.write_text("one", encoding="utf-8")
            products = [{"product_id": "p1", "title": "headphones"}]
            manifest = root / "manifest.json"
            write_search_manifest(
                manifest,
                {
                    "product_ids": ["p1"],
                    "catalog_fingerprint": search_catalog_fingerprint(products),
                    "artifact_fingerprints": {"encoder": artifact_fingerprint(artifact)},
                },
            )
            validate_search_manifest(
                manifest,
                products,
                expected_product_ids=["p1"],
                artifact_paths={"encoder": artifact},
            )
            artifact.write_text("two", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact"):
                validate_search_manifest(
                    manifest,
                    products,
                    expected_product_ids=["p1"],
                    artifact_paths={"encoder": artifact},
                )
            artifact.write_text("one", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "catalog changed"):
                validate_search_manifest(
                    manifest,
                    [{"product_id": "p1", "title": "changed"}],
                    expected_product_ids=["p1"],
                    artifact_paths={"encoder": artifact},
                )


if __name__ == "__main__":
    unittest.main()
