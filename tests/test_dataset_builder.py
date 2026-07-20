import tempfile
import unittest
from pathlib import Path

from src.preprocessing.dataset_builder import (
    build_products,
    write_processed_dataset_from_reviews,
)


def sample_review(category: str = "Automotive") -> dict:
    return {
        "user_id": "user-1",
        "product_id": "product-1",
        "rating": 5.0,
        "review_title": "Excellent product",
        "review_text": "Simple and reliable",
        "timestamp": 1_700_000_000_000,
        "helpful_vote": 0,
        "verified_purchase": True,
        "image_url": "",
        "source_category": category,
    }


class DatasetBuilderTests(unittest.TestCase):
    def test_product_uses_huggingface_source_category(self) -> None:
        products = build_products([sample_review("Health_and_Household")])
        self.assertEqual(products[0]["category"], "Health_and_Household")

    def test_common_pipeline_writes_all_tables(self) -> None:
        reviews = [sample_review(), dict(sample_review(), user_id="user-2")]
        with tempfile.TemporaryDirectory() as directory:
            summary = write_processed_dataset_from_reviews(iter(reviews), directory)
            output = Path(directory)

            self.assertEqual(summary["reviews"], 2)
            self.assertEqual(summary["reviews_by_category"], {"Automotive": 2})
            self.assertTrue((output / "products.csv").exists())
            self.assertTrue((output / "interactions.csv").exists())
            self.assertIn("source_category", (output / "reviews.csv").read_text(encoding="utf-8").splitlines()[0])


if __name__ == "__main__":
    unittest.main()
