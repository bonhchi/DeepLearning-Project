import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.amazon_reviews import (
    DEFAULT_CATEGORIES,
    huggingface_review_url,
    iter_huggingface_reviews,
    iter_reviews,
    normalize_review,
    validate_categories,
)


RAW_REVIEW = {
    "user_id": "user-1",
    "asin": "child-1",
    "parent_asin": "product-1",
    "rating": 5.0,
    "title": "Useful",
    "text": "Works well",
    "timestamp": 1_700_000_000_000,
    "helpful_vote": 2,
    "verified_purchase": True,
    "images": [],
}


class AmazonReviewReaderTests(unittest.TestCase):
    def test_normalize_review_preserves_domain_and_parent_asin(self) -> None:
        review = normalize_review(RAW_REVIEW, category="Electronics")

        self.assertEqual(review["product_id"], "product-1")
        self.assertEqual(review["source_category"], "Electronics")
        self.assertEqual(review["rating"], 5.0)

    def test_local_reader_limit_counts_valid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.jsonl"
            path.write_text(
                "not-json\n\n" + json.dumps(RAW_REVIEW) + "\n" + json.dumps(RAW_REVIEW) + "\n",
                encoding="utf-8",
            )

            reviews = list(iter_reviews(path, limit=1))

        self.assertEqual(len(reviews), 1)

    def test_huggingface_reader_streams_and_limits_each_category(self) -> None:
        calls = []

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return [dict(RAW_REVIEW, user_id=f"user-{index}") for index in range(3)]

        fake_datasets = types.SimpleNamespace(load_dataset=fake_load_dataset)
        categories = ("Automotive", "Electronics")
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            reviews = list(iter_huggingface_reviews(categories, limit_per_category=2))

        self.assertEqual(len(reviews), 4)
        self.assertEqual([row["source_category"] for row in reviews], [
            "Automotive",
            "Automotive",
            "Electronics",
            "Electronics",
        ])
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[1]["streaming"] for call in calls))
        self.assertEqual(calls[0][0][0], "json")

    def test_categories_are_explicitly_validated(self) -> None:
        self.assertEqual(validate_categories(DEFAULT_CATEGORIES), DEFAULT_CATEGORIES)
        with self.assertRaises(ValueError):
            validate_categories(["All_Beauty"])

    def test_huggingface_url_points_to_raw_review_file(self) -> None:
        url = huggingface_review_url("Health_and_Household")
        self.assertIn("McAuley-Lab/Amazon-Reviews-2023", url)
        self.assertTrue(url.endswith("raw/review_categories/Health_and_Household.jsonl"))


if __name__ == "__main__":
    unittest.main()
