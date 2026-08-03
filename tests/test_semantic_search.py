import json
import math
import tempfile
import unittest
from pathlib import Path

from src.feature_extraction.embeddings import DenseTextEncoder, TfidfEncoder
from src.semantic_search.hybrid_search import HybridSearchEngine
from src.semantic_search.lexical_index import SparseTfidfIndex
from src.semantic_search.vector_index import VectorIndex


class VectorIndexTests(unittest.TestCase):
    def test_exact_search_maps_rows_back_to_product_ids(self) -> None:
        index = VectorIndex(use_faiss=False).build(
            ["headphones", "cream", "cable"],
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
        )

        rows = index.search([1.0, 0.0], top_k=2)

        self.assertEqual([row["product_id"] for row in rows], ["headphones", "cable"])
        self.assertEqual(index.id_to_index["cream"], 1)
        self.assertEqual(index.backend, "exact")

    def test_build_rejects_duplicate_missing_and_misaligned_embeddings(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate product_id"):
            VectorIndex(use_faiss=False).build(["one", "one"], [[1.0], [2.0]])
        with self.assertRaisesRegex(ValueError, "Missing embeddings"):
            VectorIndex(use_faiss=False).build(["one", "two"], [[1.0], None])
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            VectorIndex(use_faiss=False).build(["one", "two"], [[1.0], [1.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            VectorIndex(use_faiss=False).build(["one", "two"], [[1.0]])

    def test_search_rejects_query_dimension_mismatch(self) -> None:
        index = VectorIndex(use_faiss=False).build({"one": [1.0, 0.0]})
        with self.assertRaisesRegex(ValueError, "Query dimension mismatch"):
            index.search([1.0], top_k=1)

    def test_directory_and_json_artifacts_round_trip(self) -> None:
        original = VectorIndex(use_faiss=False).build(
            {"one": [1.0, 0.0], "two": [0.0, 1.0]}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "index"
            json_path = Path(temp_dir) / "index.json"
            original.save(directory)
            original.save(json_path)

            for loaded in (
                VectorIndex.load(directory, use_faiss=False),
                VectorIndex.load(json_path, use_faiss=False),
            ):
                result = loaded.search([0.0, 1.0], top_k=1)[0]
                self.assertEqual(result["product_id"], "two")
                self.assertAlmostEqual(result["score"], 1.0, places=6)


class TextEncoderTests(unittest.TestCase):
    def test_tfidf_batch_normalization_cache_and_artifact(self) -> None:
        encoder = TfidfEncoder(max_features=20).fit(
            ["wireless headphones", "skin care cream"]
        )
        vectors = encoder.encode(["wireless headphones", "skin care cream"])

        self.assertEqual(len(vectors), 2)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vectors[0])), 1.0)
        self.assertEqual(encoder.cache_size, 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            encoder.save(temp_dir)
            loaded = TfidfEncoder.load(temp_dir)
            self.assertEqual(loaded.encode_one("wireless headphones"), vectors[0])
            self.assertEqual(loaded.vocabulary, encoder.vocabulary)

    def test_dense_fallback_is_deterministic_normalized_and_persistable(self) -> None:
        first = DenseTextEncoder(dimension=48, backend="fallback")
        second = DenseTextEncoder(dimension=48, backend="fallback")

        vector = first.encode_one("tai nghe không dây")
        self.assertEqual(vector, second.encode_one("tai nghe không dây"))
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0)
        self.assertEqual(first.cache_size, 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            first.save(temp_dir)
            loaded = DenseTextEncoder.load(temp_dir)
            self.assertEqual(loaded.encode_one("tai nghe không dây"), vector)
            self.assertEqual(loaded.requested_backend, "fallback")
            self.assertEqual(loaded.backend_name, "fallback")

            artifact = Path(temp_dir) / "dense_text_encoder.json"
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            payload["verification_vector"][0] += 0.1
            artifact.write_text(json.dumps(payload), encoding="utf-8")
            tampered = DenseTextEncoder.load(temp_dir)
            with self.assertRaisesRegex(RuntimeError, "snapshot differs"):
                tampered.encode_one("query")

    def test_sparse_tfidf_index_round_trip(self) -> None:
        documents = ["wireless headphones", "skin care cream", "wireless cable"]
        encoder = TfidfEncoder(max_features=20).fit(documents)
        index = SparseTfidfIndex().build(["h", "s", "c"], documents, encoder)
        self.assertEqual(index.search(encoder.encode_one("wireless headphones"), 2)[0]["product_id"], "h")
        with tempfile.TemporaryDirectory() as temp_dir:
            index.save(temp_dir)
            loaded = SparseTfidfIndex.load(temp_dir)
            self.assertEqual(loaded.product_ids, ["h", "s", "c"])
            self.assertEqual(loaded.search(encoder.encode_one("skin cream"), 1)[0]["product_id"], "s")


class FakeIntentClassifier:
    def predict(self, query: str) -> dict:
        return {"intent": "product_search", "confidence": 0.91}


class FakeEntityExtractor:
    def extract(self, query: str) -> dict:
        return {
            "category": {"value": "Electronics", "confidence": 0.9},
            "max_price": {"value": 100, "confidence": 0.95},
        }


class HybridSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.products = [
            {
                "product_id": "headphones",
                "title": "Wireless noise cancelling headphones",
                "description": "Bluetooth audio",
                "category": "Electronics",
                "store": "Sound",
                "price": 80,
                "features": ["wireless", "noise cancelling"],
            },
            {
                "product_id": "expensive-headphones",
                "title": "Premium wireless headphones",
                "description": "Studio audio",
                "category": "Electronics",
                "store": "Sound",
                "price": 250,
                "features": ["wireless"],
            },
            {
                "product_id": "cream",
                "title": "Hydrating skin cream",
                "description": "Daily face care",
                "category": "Beauty",
                "store": "Glow",
                "price": 20,
                "features": [],
            },
        ]

    def _engine(self, **kwargs) -> HybridSearchEngine:
        return HybridSearchEngine(
            self.products,
            dense_encoder=DenseTextEncoder(dimension=64, backend="fallback"),
            candidate_pool_size=3,
            **kwargs,
        )

    def test_all_modes_return_required_score_schema(self) -> None:
        engine = self._engine()
        for mode in ("lexical", "semantic", "hybrid"):
            row = engine.search(
                "bluetooth noise cancelling headphones", top_k=1, mode=mode
            )[0]
            self.assertEqual(row["product_id"], "headphones")
            self.assertIn("semantic_score", row)
            self.assertIn("lexical_score", row)
            self.assertIn("matched_entities", row)
            self.assertIn("filter_reason", row)

    def test_intent_entities_and_metadata_constraints_are_applied(self) -> None:
        engine = self._engine(
            intent_classifier=FakeIntentClassifier(),
            entity_extractor=FakeEntityExtractor(),
        )

        rows = engine.search("wireless headphones under 100", top_k=3)

        self.assertEqual([row["product_id"] for row in rows], ["headphones"])
        self.assertEqual(rows[0]["intent"], "product_search")
        self.assertAlmostEqual(rows[0]["intent_confidence"], 0.91)
        self.assertEqual(rows[0]["matched_entities"]["category"], ["Electronics"])
        self.assertIn("max_price=100", rows[0]["filter_reason"])
        self.assertEqual(engine.last_trace["filtered_count"], 2)
        self.assertTrue(engine.last_trace["applied_filters"])

    def test_empty_query_and_catalog_mismatch_are_rejected(self) -> None:
        engine = self._engine()
        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            engine.search("  ")

        mismatched_index = VectorIndex(use_faiss=False).build(
            ["headphones"], [[1.0, 0.0]]
        )
        with self.assertRaisesRegex(ValueError, "index/catalog mismatch"):
            self._engine(vector_index=mismatched_index)

    def test_missing_metadata_is_handled_without_key_error(self) -> None:
        class BrandExtractor:
            def extract(self, query: str) -> dict:
                return {"brand": "Acme"}

        engine = self._engine(entity_extractor=BrandExtractor())
        self.assertEqual(engine.search("Acme product", top_k=3), [])

    def test_negated_feature_excludes_products_that_contain_it(self) -> None:
        class NegatedExtractor:
            def extract(self, query: str) -> dict:
                return {
                    "category": {"value": "Electronics"},
                    "feature": [{"value": "waterproof", "negated": True}],
                }

        products = [dict(row) for row in self.products]
        products[0]["features"] = ["wireless", "waterproof"]
        engine = HybridSearchEngine(
            products,
            dense_encoder=DenseTextEncoder(dimension=64, backend="fallback"),
            entity_extractor=NegatedExtractor(),
            candidate_pool_size=3,
        )
        rows = engine.search("headphones not waterproof", top_k=3)
        self.assertEqual([row["product_id"] for row in rows], ["expensive-headphones"])
        self.assertIn("excluded_feature", rows[0]["matched_entities"])

    def test_empty_metadata_does_not_match_a_positive_feature(self) -> None:
        class FeatureExtractor:
            def extract(self, query: str) -> dict:
                return {
                    "category": {"value": "Beauty"},
                    "feature": [{"value": "waterproof"}],
                }

        engine = self._engine(entity_extractor=FeatureExtractor())
        self.assertEqual(engine.search("waterproof beauty product", top_k=3), [])

    def test_vnd_price_is_converted_to_catalog_currency(self) -> None:
        class PriceExtractor:
            def extract(self, query: str) -> dict:
                return {
                    "category": {"value": "Electronics"},
                    "max_price": {"value": 2_000_000, "currency": "VND"},
                }

        engine = self._engine(
            entity_extractor=PriceExtractor(),
            currency_rates_to_catalog={"USD": 1.0, "VND": 1.0 / 25_000.0},
        )
        rows = engine.search("tai nghe dưới 2 triệu", top_k=3)
        self.assertEqual([row["product_id"] for row in rows], ["headphones"])

    def test_metadata_match_uses_token_boundaries(self) -> None:
        self.assertFalse(HybridSearchEngine._text_matches("red", "infrared"))
        self.assertFalse(HybridSearchEngine._text_matches("less", "wireless"))
        self.assertTrue(HybridSearchEngine._text_matches("noise cancelling", "wireless noise-cancelling headphones"))

    def test_adaptive_overfetch_refills_after_hard_filter(self) -> None:
        class BrandExtractor:
            def extract(self, query):
                return {"brand": {"value": "Target"}}

        class QueryEncoder:
            dimension = 2

            def encode_one(self, text):
                return [1.0, 0.0]

            def encode(self, text):
                return self.encode_one(text)

        products = [
            {"product_id": f"other-{index}", "store": "Other", "title": "Headphones"}
            for index in range(10)
        ] + [{"product_id": "target", "store": "Target", "title": "Headphones"}]
        index = VectorIndex(use_faiss=False).build(
            [row["product_id"] for row in products],
            [[1.0, 0.0] for _ in range(10)] + [[0.5, 0.5]],
        )
        engine = HybridSearchEngine(
            products,
            vector_index=index,
            dense_encoder=QueryEncoder(),
            entity_extractor=BrandExtractor(),
            candidate_pool_size=1,
        )
        rows = engine.search("Target headphones", top_k=1, mode="semantic")
        self.assertEqual([row["product_id"] for row in rows], ["target"])
        self.assertGreater(engine.last_trace["adaptive_overfetch_rounds"], 0)


if __name__ == "__main__":
    unittest.main()
