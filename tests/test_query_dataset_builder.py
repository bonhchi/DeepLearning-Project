import csv
import tempfile
import unittest
from pathlib import Path

from src.data.query_dataset_builder import (
    QUERY_FIELDS,
    QueryDatasetValidationError,
    QueryTemplate,
    assign_deterministic_splits,
    build_query_dataset,
    save_query_dataset,
    validate_query_dataset,
)


class QueryDatasetBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.products = [
            {"product_id": "p1", "title": "Quiet Headphones", "category": "Electronics"},
            {"product_id": "p2", "title": "Running Shoes", "category": "Shoes"},
        ]

    def test_builds_schema_deduplicates_and_assigns_reproducible_splits(self) -> None:
        templates = [
            QueryTemplate("product_search", "Find {title}"),
            QueryTemplate("product_search", "  Find   {title}  "),
            QueryTemplate("comparison", "Compare {title} with other {category}"),
        ]
        first = build_query_dataset(
            self.products,
            templates=templates,
            validation_ratio=0.25,
            test_ratio=0.25,
            seed=9,
        )
        second = build_query_dataset(
            self.products,
            templates=templates,
            validation_ratio=0.25,
            test_ratio=0.25,
            seed=9,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(list(row) == QUERY_FIELDS for row in first))
        self.assertEqual(len({row["query_text"].casefold() for row in first}), len(first))
        self.assertTrue({row["split"] for row in first} <= {"train", "validation", "test"})

    def test_stratified_split_preserves_training_example_for_each_intent(self) -> None:
        rows = [
            {
                "query_id": f"q-{intent}-{index}",
                "query_text": f"query {intent} {index}",
                "intent": intent,
                "category": "test",
                "source": "unit",
                "split": "",
            }
            for intent in ("product_search", "comparison")
            for index in range(5)
        ]
        split_rows = assign_deterministic_splits(rows, validation_ratio=0.2, test_ratio=0.2)
        for intent in ("product_search", "comparison"):
            intent_splits = {row["split"] for row in split_rows if row["intent"] == intent}
            self.assertEqual(intent_splits, {"train", "validation", "test"})

    def test_save_writes_utf8_csv_and_validation_rejects_duplicate_text(self) -> None:
        rows = build_query_dataset(self.products, max_products=1)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "queries.csv"
            count = save_query_dataset(rows, target)
            with target.open(encoding="utf-8", newline="") as handle:
                stored = list(csv.DictReader(handle))
            self.assertEqual(count, len(rows))
            self.assertEqual(stored[0].keys(), dict.fromkeys(QUERY_FIELDS).keys())

        duplicate = [dict(rows[0]), dict(rows[0], query_id="another")]
        with self.assertRaises(QueryDatasetValidationError):
            validate_query_dataset(duplicate)

    def test_default_templates_are_balanced_and_grouped_by_product(self) -> None:
        products = [
            {
                "product_id": f"p{index}",
                "title": f"Unique product {index}",
                "category": "Electronics",
            }
            for index in range(10)
        ]
        rows = build_query_dataset(products, validation_ratio=0.2, test_ratio=0.2)
        counts = {
            intent: sum(row["intent"] == intent for row in rows)
            for intent in {row["intent"] for row in rows}
        }
        self.assertEqual(set(counts.values()), {20})
        for intent in counts:
            self.assertEqual(
                {row["split"] for row in rows if row["intent"] == intent},
                {"train", "validation", "test"},
            )
        splits_by_source: dict[str, set[str]] = {}
        for row in rows:
            splits_by_source.setdefault(row["source"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in splits_by_source.values()))


if __name__ == "__main__":
    unittest.main()
