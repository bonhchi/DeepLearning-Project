import unittest

from src.evaluation.metrics import (
    artifact_catalog_health,
    evaluate_recommendations,
    evaluate_serving_health,
)


class ServingHealthMetricsTests(unittest.TestCase):
    def test_ranking_metrics_return_expected_perfect_hit(self) -> None:
        metrics = evaluate_recommendations(
            recommendations_by_user={"user-1": ["relevant", "other"]},
            relevant_by_user={"user-1": {"relevant"}},
            top_k=2,
        )

        self.assertEqual(metrics["precision@2"], 0.5)
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertEqual(metrics["ndcg@2"], 1.0)
        self.assertEqual(metrics["users_evaluated"], 1)

    def test_artifact_health_detects_stale_catalog_coverage(self) -> None:
        metrics = artifact_catalog_health({"current", "stale"}, {"current", "new"})

        self.assertEqual(metrics["artifact_catalog_coverage"], 0.5)
        self.assertEqual(metrics["artifact_out_of_catalog_rate"], 0.5)

    def test_detects_duplicates_leakage_and_out_of_catalog_items(self) -> None:
        metrics = evaluate_serving_health(
            recommendations_by_user={"user-1": ["relevant", "relevant", "seen", "outside"]},
            seen_by_user={"user-1": {"seen"}},
            relevant_by_user={"user-1": {"relevant"}},
            catalog_product_ids={"seen", "relevant", "unused"},
            top_k=4,
            elapsed_ms=12.5,
        )

        self.assertEqual(metrics["catalog_coverage@4"], 0.666667)
        self.assertEqual(metrics["avg_list_size@4"], 4.0)
        self.assertEqual(metrics["duplicate_rate@4"], 0.25)
        self.assertEqual(metrics["seen_leakage_rate@4"], 0.25)
        self.assertEqual(metrics["out_of_catalog_rate@4"], 0.25)
        self.assertEqual(metrics["relevant_in_catalog_rate"], 1.0)
        self.assertEqual(metrics["latency_ms_per_user"], 12.5)


if __name__ == "__main__":
    unittest.main()
