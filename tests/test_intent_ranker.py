import unittest

from src.personalization.recommender import PersonalizedRecommender


class IntentConditionedRankerTests(unittest.TestCase):
    def setUp(self):
        products = [
            {"product_id": "semantic", "title": "A", "category": "Electronics", "average_rating": "4", "rating_number": "2"},
            {"product_id": "personal", "title": "B", "category": "Beauty", "average_rating": "4", "rating_number": "2"},
        ]
        self.recommender = PersonalizedRecommender([], products, [], {})
        self.candidates = [
            {"product_id": "semantic", "semantic_score": 1.0, "lexical_score": 1.0},
            {"product_id": "personal", "semantic_score": 0.7, "lexical_score": 0.5},
        ]

    def test_personalized_intent_uses_profile_weight(self):
        rows = self.recommender.rank_intent_candidates(
            self.candidates,
            intent="personalized_recommendation",
            user_profile={"preferred_categories": ["Beauty"]},
            top_k=2,
        )
        self.assertEqual(rows[0]["product_id"], "personal")
        self.assertIn("user_preference_score", rows[0]["score_breakdown"])

    def test_unknown_intent_falls_back_to_product_search_weights(self):
        rows = self.recommender.rank_intent_candidates(
            self.candidates, intent="unknown", top_k=1
        )
        self.assertEqual(rows[0]["product_id"], "semantic")

    def test_attribute_preferences_and_evidenced_negative_penalty(self):
        recommender = PersonalizedRecommender(
            [],
            [
                {
                    "product_id": "preferred",
                    "title": "Lightweight headset",
                    "category": "Electronics",
                    "features": '["lightweight"]',
                    "average_rating": "4",
                    "rating_number": "2",
                },
                {
                    "product_id": "disliked",
                    "title": "Waterproof headset",
                    "category": "Electronics",
                    "features": '["waterproof"]',
                    "average_rating": "4",
                    "rating_number": "2",
                },
            ],
            [],
            {},
        )
        candidates = [
            {"product_id": "disliked", "semantic_score": 0.8, "lexical_score": 0.8},
            {"product_id": "preferred", "semantic_score": 0.8, "lexical_score": 0.8},
        ]

        rows = recommender.rank_intent_candidates(
            candidates,
            intent="personalized_recommendation",
            user_profile={
                "attributes": ["lightweight"],
                "negative_preferences": ["attribute:waterproof"],
            },
            top_k=2,
        )

        self.assertEqual(rows[0]["product_id"], "preferred")
        disliked = next(row for row in rows if row["product_id"] == "disliked")
        self.assertEqual(
            disliked["profile_evidence"]["negative_matches"],
            [{"field": "attribute", "value": "waterproof"}],
        )
        self.assertEqual(disliked["profile_evidence"]["negative_penalty"], 0.4)
        self.assertLess(
            disliked["score_components"]["user_preference_score"],
            rows[0]["score_components"]["user_preference_score"],
        )

    def test_negative_preference_requires_product_evidence(self):
        rows = self.recommender.rank_intent_candidates(
            self.candidates,
            intent="personalized_recommendation",
            user_profile={"negative_preferences": ["attribute:waterproof"]},
            top_k=2,
        )

        self.assertTrue(
            all(not row["profile_evidence"]["negative_matches"] for row in rows)
        )
        self.assertTrue(
            all(row["profile_evidence"]["negative_penalty"] == 0.0 for row in rows)
        )


if __name__ == "__main__":
    unittest.main()
