import tempfile
import unittest
from pathlib import Path

from src.io_utils import json_loads_safe, read_csv_rows, write_csv_rows
from src.preprocessing.dataset_builder import (
    AMAZON_ITEM_METADATA_SOURCE,
    PRODUCT_FIELDS,
    REVIEW_FIELDS,
    REVIEW_TRAIN_METADATA_SOURCE,
    append_processed_dataset_from_reviews,
    build_interactions,
    build_products,
    write_processed_dataset_from_reviews,
)
from src.preprocessing.metadata_enricher import apply_product_metadata


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
        self.assertEqual(products[0]["metadata_source"], REVIEW_TRAIN_METADATA_SOURCE)

    def test_common_pipeline_writes_all_tables(self) -> None:
        reviews = [sample_review(), dict(sample_review(), user_id="user-2")]
        with tempfile.TemporaryDirectory() as directory:
            summary = write_processed_dataset_from_reviews(iter(reviews), directory)
            output = Path(directory)

            self.assertEqual(summary["reviews"], 2)
            self.assertEqual(summary["reviews_by_category"], {"Automotive": 2})
            self.assertTrue((output / "products.csv").exists())
            self.assertTrue((output / "interactions.csv").exists())
            self.assertTrue((output / "dataset_manifest.json").exists())
            self.assertIn("source_category", (output / "reviews.csv").read_text(encoding="utf-8").splitlines()[0])

    def test_product_text_and_rating_only_use_training_reviews(self) -> None:
        reviews = [
            dict(sample_review(), product_id="train", timestamp=1, review_title="Training title"),
            dict(sample_review(), product_id="val", timestamp=2, review_title="Validation secret"),
            dict(sample_review(), product_id="test", timestamp=3, review_title="Test secret"),
        ]
        interactions = build_interactions(reviews)
        train_keys = {
            (row["user_id"], row["product_id"], int(row["timestamp"]))
            for row in interactions
            if row["split"] == "train"
        }
        products = {row["product_id"]: row for row in build_products(reviews, train_keys)}
        self.assertEqual(products["train"]["title"], "Training title")
        self.assertEqual(products["test"]["title"], "Amazon Item test")
        self.assertNotIn("secret", products["test"]["description"].casefold())
        self.assertEqual(products["test"]["rating_number"], 0)

    def test_events_after_holdout_cutoff_never_remain_in_train(self) -> None:
        reviews = [
            dict(sample_review(), product_id="early", timestamp=100, rating=5.0),
            dict(sample_review(), product_id="target", timestamp=200, rating=5.0),
            dict(sample_review(), product_id="future-negative", timestamp=300, rating=1.0),
        ]
        interactions = build_interactions(reviews)
        by_product = {row["product_id"]: row for row in interactions}
        self.assertEqual(by_product["early"]["split"], "train")
        self.assertEqual(by_product["target"]["split"], "test")
        self.assertEqual(by_product["future-negative"]["split"], "test")
        latest_train = max(
            int(row["timestamp"]) for row in interactions if row["split"] == "train"
        )
        earliest_holdout = min(
            int(row["timestamp"]) for row in interactions if row["split"] != "train"
        )
        self.assertLess(latest_train, earliest_holdout)

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

    def test_append_rebuilds_catalog_when_old_train_event_moves_to_test(self) -> None:
        original = dict(
            sample_review(),
            timestamp=100,
            review_title="Heldout secret title",
            review_text="heldoutsecret must disappear",
            rating=5.0,
        )
        earlier = dict(
            sample_review(),
            timestamp=50,
            review_title="Earlier safe title",
            review_text="training content only",
            rating=4.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            write_processed_dataset_from_reviews([original], directory)
            append_processed_dataset_from_reviews([earlier], directory)
            output = Path(directory)
            product = read_csv_rows(output / "products.csv")[0]
            interactions = read_csv_rows(output / "interactions.csv")

        moved_event = next(row for row in interactions if row["timestamp"] == "100")
        self.assertEqual(moved_event["split"], "test")
        self.assertEqual(product["title"], "Earlier safe title")
        self.assertNotIn("heldoutsecret", product["description"].casefold())
        self.assertNotIn("heldoutsecret", product["features"].casefold())
        self.assertEqual(product["average_rating"], "4.0")
        self.assertEqual(product["rating_number"], "1")

    def test_append_restores_only_fields_with_amazon_metadata_provenance(self) -> None:
        original = dict(
            sample_review(),
            timestamp=100,
            review_title="Old review title",
            review_text="old heldout description",
            rating=5.0,
        )
        earlier = dict(
            sample_review(),
            timestamp=50,
            review_title="New training title",
            review_text="new training description",
            rating=4.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_processed_dataset_from_reviews([original], output)
            products = read_csv_rows(output / "products.csv")
            apply_product_metadata(products[0], {"title": "Independent catalog title"})
            write_csv_rows(output / "products.csv", products, PRODUCT_FIELDS)

            append_processed_dataset_from_reviews([earlier], output)
            product = read_csv_rows(output / "products.csv")[0]

        provenance = json_loads_safe(product["metadata_provenance"], {})
        self.assertEqual(product["metadata_source"], AMAZON_ITEM_METADATA_SOURCE)
        self.assertEqual(product["title"], "Independent catalog title")
        self.assertEqual(provenance["title"], AMAZON_ITEM_METADATA_SOURCE)
        self.assertEqual(product["description"], "new training description")
        self.assertEqual(product["average_rating"], "4.0")

    def test_append_migrates_legacy_csv_without_provenance_columns(self) -> None:
        original = dict(sample_review(), timestamp=100, review_title="Legacy title")
        earlier = dict(sample_review(), timestamp=50, review_title="Safe rebuilt title")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_processed_dataset_from_reviews([original], output)
            write_csv_rows(
                output / "products.csv",
                read_csv_rows(output / "products.csv"),
                PRODUCT_FIELDS[:-2],
            )
            write_csv_rows(
                output / "reviews.csv",
                read_csv_rows(output / "reviews.csv"),
                [field for field in REVIEW_FIELDS if field not in {"verified_purchase", "image_url"}],
            )

            append_processed_dataset_from_reviews([earlier], output)
            product = read_csv_rows(output / "products.csv")[0]

        self.assertEqual(product["title"], "Safe rebuilt title")
        self.assertEqual(product["metadata_source"], REVIEW_TRAIN_METADATA_SOURCE)
        self.assertTrue(product["metadata_provenance"])


if __name__ == "__main__":
    unittest.main()
