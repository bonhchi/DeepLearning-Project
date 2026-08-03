import unittest

from src.nlp.entity_extractor import EntityExtractor
from src.nlp.query_rewriter import QueryRewriter, rewrite_query


class QueryRewriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = EntityExtractor()
        self.rewriter = QueryRewriter(self.extractor)

    def test_empty_query_is_not_replaced_with_profile_preferences(self) -> None:
        result = self.rewriter.rewrite(
            "",
            intent="personalized_recommendation",
            user_profile={"preferred_brands": ["Sony"]},
        )

        self.assertEqual(
            result,
            {
                "original_query": "",
                "rewritten_query": "",
                "added_preferences": [],
                "ignored_preferences": [],
            },
        )

    def test_missing_profile_is_supported(self) -> None:
        result = self.rewriter.rewrite(
            "wireless headphones", intent="product_search", user_profile=None
        )

        self.assertEqual(result["rewritten_query"], "wireless headphones")
        self.assertEqual(result["added_preferences"], [])
        self.assertEqual(result["ignored_preferences"], [])

    def test_unknown_intent_preserves_query_and_does_not_personalize(self) -> None:
        result = self.rewriter.rewrite(
            "headphones",
            intent="not_a_real_intent",
            user_profile={"preferred_brands": ["Sony"]},
        )

        self.assertEqual(result["rewritten_query"], "headphones")
        self.assertEqual(result["added_preferences"], [])
        self.assertEqual(result["ignored_preferences"][0]["reason"], "unknown_intent")

    def test_adds_non_conflicting_profile_preferences(self) -> None:
        result = self.rewriter.rewrite(
            "Tôi cần tai nghe",
            intent="personalized_recommendation",
            user_profile={
                "preferred_brands": ["Sony"],
                "preferred_features": ["wireless"],
            },
        )

        self.assertIn("thương hiệu Sony", result["rewritten_query"])
        self.assertIn("không dây", result["rewritten_query"])
        self.assertEqual(
            [(row["field"], row["value"]) for row in result["added_preferences"]],
            [("brand", "Sony"), ("feature", "wireless")],
        )

    def test_explicit_brand_and_color_override_profile(self) -> None:
        query = "Tôi cần tai nghe Sony màu đen"
        result = self.rewriter.rewrite(
            query,
            intent="personalized_recommendation",
            user_profile={
                "preferred_brands": ["Samsung"],
                "preferred_colors": ["white"],
            },
        )

        self.assertEqual(result["rewritten_query"], query)
        self.assertEqual(result["added_preferences"], [])
        self.assertEqual(
            {row["field"] for row in result["ignored_preferences"]},
            {"brand", "color"},
        )
        self.assertTrue(
            all(
                row["reason"] == "conflicts_with_explicit_query_constraint"
                for row in result["ignored_preferences"]
            )
        )

    def test_explicit_price_constraint_overrides_profile_price_range(self) -> None:
        result = self.rewriter.rewrite(
            "headphones less than $100",
            intent="product_search",
            user_profile={
                "price_range": {"min": 200, "max": 400, "currency": "USD"}
            },
        )

        self.assertNotIn("$200", result["rewritten_query"])
        self.assertEqual(result["added_preferences"], [])
        self.assertEqual(
            result["ignored_preferences"][0]["reason"],
            "query_price_constraint_takes_priority",
        )

    def test_opposite_feature_is_not_added(self) -> None:
        result = self.rewriter.rewrite(
            "wired headphones",
            intent="product_search",
            user_profile={"preferred_features": ["wireless"]},
        )

        self.assertEqual(result["rewritten_query"], "wired headphones")
        self.assertEqual(result["added_preferences"], [])
        self.assertEqual(
            result["ignored_preferences"][0]["reason"],
            "conflicts_with_explicit_query_constraint",
        )

    def test_availability_intent_uses_rule_template(self) -> None:
        result = rewrite_query("Sony WH-1000XM5", intent="availability_check")

        self.assertEqual(result["original_query"], "Sony WH-1000XM5")
        self.assertIn("in stock", result["rewritten_query"])

    def test_accepts_entity_envelope_schema(self) -> None:
        entities = {"entities": self.extractor.extract("red shoes")}
        result = self.rewriter.rewrite(
            "red shoes",
            intent="product_search",
            entities=entities,
            user_profile={"preferred_colors": ["black"]},
        )

        self.assertEqual(result["rewritten_query"], "red shoes")
        self.assertEqual(result["ignored_preferences"][0]["field"], "color")

    def test_structured_negative_preference_is_an_explicit_exclusion(self) -> None:
        result = self.rewriter.rewrite(
            "headphones",
            intent="personalized_recommendation",
            user_profile={"negative_preferences": ["brand:Sony"]},
        )

        self.assertIn("exclude brand Sony", result["rewritten_query"])
        self.assertEqual(
            result["added_preferences"][0]["reason"],
            "from_user_profile_negative",
        )

    def test_explicit_query_overrides_historical_negative_preference(self) -> None:
        result = self.rewriter.rewrite(
            "Sony headphones",
            intent="personalized_recommendation",
            user_profile={"negative_preferences": ["brand:Sony"]},
        )

        self.assertEqual(result["rewritten_query"], "Sony headphones")
        self.assertEqual(result["added_preferences"], [])
        self.assertEqual(
            result["ignored_preferences"][0]["reason"],
            "explicit_query_overrides_negative_profile",
        )

    def test_inconsistent_positive_and_negative_profile_does_not_add_both(self) -> None:
        result = self.rewriter.rewrite(
            "headphones",
            intent="personalized_recommendation",
            user_profile={
                "preferred_brands": ["Sony"],
                "negative_preferences": ["brand:Sony"],
            },
        )

        self.assertNotIn("; brand Sony;", result["rewritten_query"])
        self.assertIn("exclude brand Sony", result["rewritten_query"])
        self.assertEqual(
            result["ignored_preferences"][0]["reason"],
            "conflicts_with_negative_profile_preference",
        )

    def test_unstructured_negative_review_term_is_not_injected(self) -> None:
        result = self.rewriter.rewrite(
            "headphones",
            intent="personalized_recommendation",
            user_profile={"negative_preferences": ["review_term:cheap"]},
        )

        self.assertEqual(result["rewritten_query"], "headphones")
        self.assertEqual(
            result["ignored_preferences"][0]["reason"],
            "unsafe_unstructured_negative_preference",
        )


if __name__ == "__main__":
    unittest.main()
