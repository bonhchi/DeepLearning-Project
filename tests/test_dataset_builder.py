import tempfile
import unittest
from pathlib import Path

from src.preprocessing.dataset_builder import (
    append_processed_dataset_from_reviews,
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

    def test_append_preserves_catalog_and_adds_local_fashion_subcategory(self) -> None:
        fashion_review = dict(
            sample_review(""),
            user_id="fashion-user",
            product_id="shirt-1",
            review_title="Comfortable cotton shirt",
            review_text="Soft t-shirt tee for everyday wear",
            timestamp=1_700_000_000_001,
        )
        with tempfile.TemporaryDirectory() as directory:
            write_processed_dataset_from_reviews([sample_review()], directory)

            summary = append_processed_dataset_from_reviews([fashion_review], directory)
            products = (Path(directory) / "products.csv").read_text(encoding="utf-8")

            self.assertEqual(summary["appended_reviews"], 1)
            self.assertIn("Automotive", products)
            self.assertIn("tops", products)


if __name__ == "__main__":
    unittest.main()
