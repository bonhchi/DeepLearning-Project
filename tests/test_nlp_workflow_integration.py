import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from main import (
    _load_search_catalog,
    _sanitize_audit_value,
    run_ablation,
    run_audit_search,
    run_evaluate_intent,
    run_evaluate_search,
    run_index_semantic,
    run_pool_qrels,
    run_prepare_queries,
    run_train_intent,
)
from src.config import ProjectConfig, ensure_project_dirs
from src.data.qrels_builder import write_qrels
from src.io_utils import read_csv_rows, write_csv_rows
from src.preprocessing.dataset_builder import write_dataset_manifest


class NlpWorkflowIntegrationTests(unittest.TestCase):
    def test_search_catalog_loader_returns_inventory_enriched_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ProjectConfig(Path(directory))
            ensure_project_dirs(config)
            write_csv_rows(
                config.products_path,
                [{"product_id": "p", "title": "Product"}],
                ["product_id", "title"],
            )
            write_csv_rows(
                config.business_context_path,
                [{"product_id": "p", "inventory_score": "0.9"}],
                ["product_id", "inventory_score"],
            )
            rows = _load_search_catalog(config)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["available"])

    def test_audit_redacts_direct_contact_identifiers(self):
        value = _sanitize_audit_value(
            {"query": "email me at person@example.com or +84 912 345 678"}
        )
        self.assertNotIn("person@example.com", value["query"])
        self.assertNotIn("912 345 678", value["query"])

    def test_small_catalog_runs_from_queries_to_audit_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ProjectConfig(Path(directory))
            ensure_project_dirs(config)
            products = [
                {
                    "product_id": f"p{index}",
                    "title": f"Headphones model {index}",
                    "category": "Electronics",
                    "store": f"Brand{index % 2}",
                    "price": str(40 + index),
                    "average_rating": "4.5",
                    "rating_number": str(10 + index),
                    "description": "wireless noise cancelling headphones",
                    "features": '["wireless", "noise cancelling"]',
                    "image_url": "",
                    "preferred_price_range": "budget",
                    "available": "true",
                }
                for index in range(6)
            ]
            write_csv_rows(config.products_path, products, list(products[0]))
            write_csv_rows(config.users_path, [{"user_id": "u1"}], ["user_id"])
            interactions = [
                {
                    "user_id": "u1",
                    "product_id": "p0",
                    "event_weight": "5",
                    "rating": "5",
                    "timestamp": "1700000000000",
                    "label": "1",
                    "split": "train",
                },
                {
                    "user_id": "u1",
                    "product_id": "p4",
                    "event_weight": "5",
                    "rating": "5",
                    "timestamp": "1700000001000",
                    "label": "1",
                    "split": "test",
                },
            ]
            write_csv_rows(
                config.interactions_path,
                interactions,
                ["user_id", "product_id", "event_weight", "rating", "timestamp", "label", "split"],
            )
            write_dataset_manifest(config.processed_dir)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                query_summary = run_prepare_queries(
                    argparse.Namespace(
                        category="Electronics",
                        max_products=6,
                        validation_ratio=0.1,
                        test_ratio=0.1,
                        seed=42,
                        qrels_per_query=3,
                    ),
                    config,
                )
                review_rows = read_csv_rows(config.qrels_review_path)
                review_rows[0]["reviewed"] = "true"
                write_qrels(config.qrels_review_path, review_rows, review=True)
                classifier = run_train_intent(
                    argparse.Namespace(backend="python", max_features=256, max_iter=8),
                    config,
                )
                intent_metrics = run_evaluate_intent(
                    argparse.Namespace(split="test"), config
                )
                index_summary = run_index_semantic(
                    argparse.Namespace(
                        category="Electronics",
                        max_products=6,
                        batch_size=4,
                        dense_dim=24,
                        lexical_features=64,
                        dense_backend="fallback",
                        model_name="offline-test",
                        allow_model_download=False,
                    ),
                    config,
                )
                pool_summary = run_pool_qrels(
                    argparse.Namespace(top_k=3, qrels_per_query=13), config
                )
                review_rows = read_csv_rows(config.qrels_review_path)
                for row in review_rows:
                    row["reviewed"] = "true"
                write_qrels(config.qrels_review_path, review_rows, review=True)
                audit = run_audit_search(
                    argparse.Namespace(
                        query="Find wireless headphones under $100",
                        user_id="u1",
                        product_id="",
                        top_k=3,
                        mode="hybrid",
                        intent="product_search",
                    ),
                    config,
                )
                search_metrics = run_evaluate_search(
                    argparse.Namespace(top_k=3, allow_unreviewed_qrels=False),
                    config,
                )
                ablation = run_ablation(
                    argparse.Namespace(top_k=3, allow_unreviewed_qrels=False),
                    config,
                )

            self.assertEqual(query_summary["products_used"], 6)
            self.assertGreater(query_summary["user_aware_personalized_queries"], 0)
            user_aware_queries = [
                row for row in read_csv_rows(config.query_dataset_path)
                if row["source"].startswith("user_holdout:u1:p4")
            ]
            self.assertTrue(user_aware_queries)
            self.assertTrue(
                all("Headphones model 4" not in row["query_text"] for row in user_aware_queries)
            )
            self.assertEqual(set(classifier.classes_), {
                "product_search", "need_based_search", "similar_product_search",
                "personalized_recommendation", "availability_check", "comparison",
            })
            self.assertGreater(intent_metrics["sample_count"], 0)
            self.assertEqual(index_summary["indexed_products"], 6)
            self.assertGreater(pool_summary["queries_pooled"], 0)
            self.assertGreaterEqual(pool_summary["manual_reviews_preserved"], 1)
            self.assertEqual(audit["detected_intent"], "product_search")
            self.assertTrue(audit["top_results"])
            self.assertEqual(
                set(search_metrics),
                {"tfidf", "dense_semantic", "semantic_intent", "semantic_intent_personalized"},
            )
            self.assertGreater(
                search_metrics["tfidf"]["aggregate"]["queries_evaluated"], 0
            )
            self.assertTrue(config.search_audit_path.exists())
            self.assertEqual(len(ablation["results"]), 5)
            self.assertTrue(config.ablation_report_path.exists())


if __name__ == "__main__":
    unittest.main()
