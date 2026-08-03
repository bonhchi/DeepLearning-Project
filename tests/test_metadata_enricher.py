import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.io_utils import json_loads_safe, read_csv_rows, write_csv_rows
from src.preprocessing.dataset_builder import AMAZON_ITEM_METADATA_SOURCE, PRODUCT_FIELDS
from src.preprocessing.metadata_enricher import enrich_product_images, infer_metadata_categories


class MetadataEnricherTests(unittest.TestCase):
    def test_local_fashion_subcategory_maps_to_amazon_fashion_metadata(self) -> None:
        self.assertEqual(
            infer_metadata_categories([{"category": "tops"}]),
            ["Amazon_Fashion"],
        )

    def test_enrichment_adds_real_catalog_fields(self) -> None:
        products = [
            {
                "product_id": "P1",
                "title": "Review title",
                "category": "tops",
                "store": "synthetic",
                "price": 10,
                "average_rating": 4,
                "rating_number": 1,
                "description": "review",
                "features": [],
                "image_url": "",
                "preferred_price_range": "budget",
            }
        ]
        metadata = {
            "product_id": "P1",
            "title": "Real product title",
            "store": "Real store",
            "price": 39.99,
            "average_rating": 4.8,
            "rating_number": 120,
            "description": "Catalog description",
            "features": ["soft cotton"],
            "image_url": "https://example.com/main.jpg",
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.csv"
            write_csv_rows(path, products, PRODUCT_FIELDS)
            with patch(
                "src.preprocessing.metadata_enricher.iter_huggingface_product_metadata",
                return_value=iter([metadata]),
            ):
                summary = enrich_product_images(path, categories=["Amazon_Fashion"])
            enriched = read_csv_rows(path)[0]

        self.assertEqual(summary["images_added"], 1)
        self.assertEqual(summary["missing_images_after"], 0)
        self.assertEqual(enriched["image_url"], "https://example.com/main.jpg")
        self.assertEqual(enriched["title"], "Real product title")
        self.assertEqual(enriched["price"], "39.99")
        self.assertEqual(enriched["metadata_source"], AMAZON_ITEM_METADATA_SOURCE)
        provenance = json_loads_safe(enriched["metadata_provenance"], {})
        self.assertEqual(provenance["title"], AMAZON_ITEM_METADATA_SOURCE)
        self.assertEqual(provenance["image_url"], AMAZON_ITEM_METADATA_SOURCE)


if __name__ == "__main__":
    unittest.main()
