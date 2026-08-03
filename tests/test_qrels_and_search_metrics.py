import tempfile
import unittest
from pathlib import Path

from src.data.qrels_builder import (
    build_qrels,
    merge_review_queue,
    validate_qrels,
    write_qrels,
)
from src.evaluation.search_metrics import evaluate_search_configuration, save_search_metrics


class QrelsAndSearchMetricsTests(unittest.TestCase):
    def test_holdout_requires_manual_review(self):
        queries = [{"query_id": "q1", "query_text": "red shoe", "category": "shoes", "intent": "product_search", "split": "test"}]
        products = [{"product_id": "p1", "title": "red shoe", "category": "shoes"}]
        qrels, queue = build_qrels(queries, products)
        self.assertEqual(len(queue), 1)
        metrics = evaluate_search_configuration(
            queries, qrels, lambda query, k, context: ["p1"], top_k=5
        )
        self.assertEqual(metrics["aggregate"]["queries_evaluated"], 0)
        self.assertEqual(metrics["aggregate"]["pending_manual_review"], 1)

    def test_reviewed_qrel_produces_perfect_metrics(self):
        queries = [{"query_id": "q1", "query_text": "red shoe", "intent": "product_search", "split": "test"}]
        qrels = [{"query_id": "q1", "product_id": "p1", "relevance": "2", "reviewed": "true"}]
        metrics = evaluate_search_configuration(
            queries, qrels, lambda query, k, context: ["p1"], top_k=1
        )
        self.assertEqual(metrics["aggregate"]["ndcg@1"], 1.0)
        self.assertEqual(metrics["per_intent"]["product_search"]["recall@1"], 1.0)

    def test_missing_review_flag_is_not_treated_as_human_judgment(self):
        queries = [{"query_id": "q1", "query_text": "shoe", "intent": "product_search", "split": "test"}]
        qrels = [{"query_id": "q1", "product_id": "p1", "relevance": "2"}]
        metrics = evaluate_search_configuration(
            queries, qrels, lambda query, k, context: ["p1"], top_k=1
        )
        self.assertEqual(metrics["aggregate"]["queries_evaluated"], 0)
        self.assertEqual(metrics["aggregate"]["pending_manual_review"], 1)

    def test_reviewed_query_with_no_positive_qrel_counts_as_zero(self):
        queries = [{"query_id": "q1", "query_text": "shoe", "intent": "product_search", "split": "test"}]
        qrels = [{"query_id": "q1", "product_id": "p1", "relevance": "0", "reviewed": "true"}]
        metrics = evaluate_search_configuration(
            queries, qrels, lambda query, k, context: ["p1"], top_k=1
        )
        self.assertEqual(metrics["aggregate"]["queries_evaluated"], 1)
        self.assertEqual(metrics["aggregate"]["ndcg@1"], 0.0)

    def test_partially_reviewed_query_is_not_evaluated(self):
        queries = [{"query_id": "q1", "query_text": "shoe", "intent": "product_search", "split": "test"}]
        qrels = [
            {"query_id": "q1", "product_id": "p1", "relevance": "2", "reviewed": "true"},
            {"query_id": "q1", "product_id": "p2", "relevance": "0", "reviewed": "false"},
        ]
        metrics = evaluate_search_configuration(
            queries, qrels, lambda query, k, context: ["p1"], top_k=1
        )
        self.assertEqual(metrics["aggregate"]["queries_evaluated"], 0)
        self.assertEqual(metrics["aggregate"]["queries_pending_manual_review"], 1)

    def test_holdout_query_without_qrels_is_not_silently_dropped(self):
        queries = [{"query_id": "q1", "query_text": "rare query", "intent": "need_based_search", "split": "test"}]
        official = evaluate_search_configuration(
            queries, [], lambda query, k, context: [], top_k=1
        )
        self.assertEqual(official["aggregate"]["queries_evaluated"], 0)
        self.assertEqual(official["aggregate"]["queries_without_qrels"], 1)
        self.assertEqual(official["aggregate"]["queries_pending_manual_review"], 1)
        smoke = evaluate_search_configuration(
            queries, [], lambda query, k, context: [], top_k=1,
            require_reviewed_holdout=False,
        )
        self.assertEqual(smoke["aggregate"]["queries_evaluated"], 1)
        self.assertEqual(smoke["aggregate"]["ndcg@1"], 0.0)

    def test_rebuild_preserves_completed_manual_judgment(self):
        fresh = [{"query_id": "q1", "product_id": "p1", "relevance": 0, "reviewed": "false"}]
        existing = [{"query_id": "q1", "product_id": "p1", "relevance": 2, "reviewed": "true"}]
        merged, count = merge_review_queue(fresh, existing)
        self.assertEqual(count, 1)
        self.assertEqual(merged[0]["relevance"], 2)
        self.assertEqual(merged[0]["source"], "manual_review")

    def test_similar_query_excludes_anchor_and_availability_uses_stock_flag(self):
        products = [
            {"product_id": "anchor", "title": "red running shoe", "category": "shoes", "available": "false"},
            {"product_id": "other", "title": "red running trainer", "category": "shoes", "available": "true"},
        ]
        similar_query = [{
            "query_id": "similar", "query_text": "similar red running shoe",
            "category": "shoes", "intent": "similar_product_search",
            "source": "product_metadata:anchor", "split": "test",
        }]
        similar_qrels, _ = build_qrels(
            similar_query, products, candidate_pools={"similar": ["anchor", "other"]}
        )
        self.assertNotIn("anchor", {row["product_id"] for row in similar_qrels})

        availability_query = [{
            "query_id": "stock", "query_text": "is red running shoe available",
            "category": "shoes", "intent": "availability_check",
            "source": "product_metadata:anchor", "split": "test",
        }]
        availability_qrels, _ = build_qrels(availability_query, products)
        anchor = next(row for row in availability_qrels if row["product_id"] == "anchor")
        self.assertEqual(anchor["relevance"], 0)
        self.assertEqual(anchor["source"], "unavailable_source_product")

    def test_duplicate_qrel_is_rejected(self):
        rows = [
            {"query_id": "q", "product_id": "p", "relevance": 1},
            {"query_id": "q", "product_id": "p", "relevance": 2},
        ]
        with self.assertRaises(ValueError):
            validate_qrels(rows)

    def test_search_metric_csv_contains_per_intent_rows(self):
        metrics = {
            "dense": {
                "aggregate": {"ndcg@5": 0.5, "queries_evaluated": 1},
                "per_intent": {"product_search": {"ndcg@5": 0.5, "queries_evaluated": 1}},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "metrics.json"
            csv_path = Path(directory) / "metrics.csv"
            save_search_metrics(json_path, csv_path, metrics)
            content = csv_path.read_text(encoding="utf-8")
            self.assertIn("per_intent", content)
            self.assertIn("product_search", content)


if __name__ == "__main__":
    unittest.main()
