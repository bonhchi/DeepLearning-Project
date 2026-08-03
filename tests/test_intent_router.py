import unittest

from src.nlp.entity_extractor import EntityExtractor
from src.nlp.query_rewriter import QueryRewriter
from src.personalization.intent_router import IntentRouter


class FakeEngine:
    def __init__(self):
        self.last_query = ""

    def search(self, query, top_k=10, mode="hybrid"):
        self.last_query = query
        return [
            {"product_id": "out", "score": 0.9, "available": "false"},
            {"product_id": "in", "score": 0.8, "available": "true"},
        ][:top_k]


class FakeClassifier:
    def __init__(self, intent):
        self.intent = intent

    def predict(self, query):
        return {"intent": self.intent, "confidence": 0.9}


class FakeRecommender:
    seen_by_user = {"u": {"out"}}

    def rank_intent_candidates(self, candidates, **kwargs):
        return candidates[: kwargs["top_k"]]


class AnchorRecommender(FakeRecommender):
    products = {
        "source": {
            "product_id": "source",
            "title": "Bose QuietComfort 45",
        }
    }

    def __init__(self):
        self.similar_source = ""

    def similar_products(self, product_id, top_k=10):
        self.similar_source = product_id
        return [
            {
                "product_id": "alternative",
                "score": 0.8,
                "score_breakdown": {"Text": 70.0},
            }
        ]


class SubsetEngine(FakeEngine):
    products = {
        "source": {"product_id": "source"},
        "allowed": {"product_id": "allowed"},
    }


class FullCatalogAnchorRecommender(AnchorRecommender):
    def similar_products(self, product_id, top_k=10):
        self.similar_source = product_id
        return [
            {"product_id": "outside-index", "score": 0.99},
            {"product_id": "allowed", "score": 0.8},
        ]


class BoseOnlyEngine(FakeEngine):
    def search(self, query, top_k=10, mode="hybrid"):
        self.last_query = query
        return [
            {
                "product_id": "bose-only",
                "brand": "Bose",
                "title": "Bose headphones",
                "score": 0.8,
                "available": "true",
            }
        ][:top_k]


class IntentExpansionEngine(FakeEngine):
    def search(self, query, top_k=10, mode="hybrid"):
        self.last_query = query
        if "similar alternatives" in query:
            return [{"product_id": "expanded", "score": 0.9}]
        return [{"product_id": "literal", "score": 0.5}]


class MultiBrandEngine(FakeEngine):
    def search(self, query, top_k=10, mode="hybrid"):
        self.last_query = query
        normalized = query.casefold()
        # Mimic HybridSearch's scalar-brand hard filter: a query containing both
        # anchors would return only the first one.
        if "bose" in normalized:
            return [{"product_id": "bose", "store": "Bose", "score": 0.9}]
        if "sony" in normalized:
            return [{"product_id": "sony", "store": "Sony", "score": 0.85}]
        return []


class IntentRouterTests(unittest.TestCase):
    def test_availability_strategy_applies_hard_filter(self):
        router = IntentRouter(FakeEngine(), intent_classifier=FakeClassifier("availability_check"))
        output = router.route("Còn hàng không?", top_k=5)
        self.assertEqual([row["product_id"] for row in output["results"]], ["in"])
        self.assertEqual(output["strategy"], "availability_check")

    def test_unknown_intent_uses_product_search(self):
        router = IntentRouter(FakeEngine(), intent_classifier=FakeClassifier("unknown"))
        output = router.route("headphones", top_k=1)
        self.assertEqual(output["detected_intent"], "product_search")

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            IntentRouter(FakeEngine()).route("  ")

    def test_seen_products_are_removed_for_personalized_recommendation(self):
        router = IntentRouter(FakeEngine(), recommender=FakeRecommender())
        output = router.route(
            "headphones",
            user_id="u",
            top_k=5,
            intent="personalized_recommendation",
        )
        self.assertEqual([row["product_id"] for row in output["results"]], ["in"])
        self.assertEqual(output["seen_filtered_count"], 1)

    def test_seen_products_are_kept_for_exact_product_search(self):
        router = IntentRouter(FakeEngine(), recommender=FakeRecommender())
        output = router.route("headphones", user_id="u", top_k=5)

        self.assertEqual(
            [row["product_id"] for row in output["results"]], ["out", "in"]
        )
        self.assertEqual(output["seen_filtered_count"], 0)

    def test_profile_rewrite_remains_soft_and_does_not_replace_retrieval_query(self):
        engine = BoseOnlyEngine()
        router = IntentRouter(
            engine,
            entity_extractor=EntityExtractor(),
            query_rewriter=QueryRewriter(),
            recommender=FakeRecommender(),
        )
        output = router.route(
            "headphones",
            intent="personalized_recommendation",
            user_profile={"preferred_brands": ["Sony"]},
            top_k=5,
        )

        self.assertEqual(engine.last_query, "headphones")
        self.assertEqual(output["retrieval_query"], "headphones")
        self.assertIn("brand Sony", output["rewritten_query"])
        self.assertEqual(
            [row["product_id"] for row in output["results"]], ["bose-only"]
        )

    def test_intent_only_rewrite_is_used_for_retrieval(self):
        engine = IntentExpansionEngine()
        router = IntentRouter(engine, query_rewriter=QueryRewriter())

        output = router.route(
            "headphones",
            intent="similar_product_search",
            top_k=5,
        )

        self.assertIn("similar alternatives", engine.last_query)
        self.assertEqual(output["retrieval_query"], engine.last_query)
        self.assertEqual(output["results"][0]["product_id"], "expanded")

    def test_comparison_retrieves_each_named_brand_anchor(self):
        router = IntentRouter(MultiBrandEngine(), query_rewriter=QueryRewriter())

        output = router.route(
            "Compare Bose and Sony headphones",
            intent="comparison",
            top_k=2,
        )

        self.assertEqual(
            [row["product_id"] for row in output["results"]], ["bose", "sony"]
        )
        self.assertEqual(
            {row["comparison_anchor"] for row in output["results"]},
            {"Bose", "Sony"},
        )

    def test_similar_query_resolves_exact_catalog_title_as_item_anchor(self):
        recommender = AnchorRecommender()
        router = IntentRouter(
            FakeEngine(),
            query_rewriter=QueryRewriter(),
            recommender=recommender,
        )

        output = router.route(
            "Show me products similar to Bose QuietComfort 45",
            intent="similar_product_search",
            top_k=3,
        )

        self.assertEqual(recommender.similar_source, "source")
        self.assertEqual(output["source_product_id"], "source")
        self.assertEqual(output["results"][0]["product_id"], "alternative")

    def test_similar_results_are_constrained_to_search_index_subset(self):
        recommender = FullCatalogAnchorRecommender()
        router = IntentRouter(SubsetEngine(), recommender=recommender)
        output = router.route(
            "similar to source",
            product_id="source",
            intent="similar_product_search",
            top_k=3,
        )
        self.assertEqual([row["product_id"] for row in output["results"]], ["allowed"])


if __name__ == "__main__":
    unittest.main()
