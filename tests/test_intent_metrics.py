import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.intent_metrics import compute_intent_metrics, save_intent_metrics


class IntentMetricsTests(unittest.TestCase):
    def test_computes_macro_per_class_and_confusion_metrics(self) -> None:
        metrics = compute_intent_metrics(
            ["search", "search", "compare", "compare"],
            ["search", "compare", "compare", "compare"],
        )

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["labels"], ["compare", "search"])
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertEqual(metrics["per_class"]["search"]["recall"], 0.5)
        self.assertEqual(metrics["macro_recall"], 0.75)

    def test_saves_json_and_tidy_csv_with_confusion_rows(self) -> None:
        metrics = compute_intent_metrics(["a", "b"], ["a", "a"])
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "intent_metrics.json"
            csv_path = Path(directory) / "intent_metrics.csv"
            save_intent_metrics(metrics, json_path, csv_path)

            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["accuracy"], 0.5)
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["section"] == "summary" for row in rows))
            self.assertTrue(any(row["section"] == "confusion_matrix" for row in rows))

    def test_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            compute_intent_metrics(["a"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()

