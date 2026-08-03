import unittest

from src.personalization.user_profile import UserProfileBuilder


class UserProfileBuilderTests(unittest.TestCase):
    def test_uses_train_only_and_applies_recent_behavior(self):
        products = [
            {"product_id": "old", "category": "Electronics", "store": "Old", "price": "50"},
            {"product_id": "new", "category": "Beauty", "store": "New", "price": "100"},
            {"product_id": "leak", "category": "Automotive", "store": "Leak", "price": "999"},
        ]
        interactions = [
            {"user_id": "u", "product_id": "old", "split": "train", "label": "1", "event_weight": "1", "timestamp": "1000"},
            {"user_id": "u", "product_id": "new", "split": "train", "label": "1", "event_weight": "1", "timestamp": str(1000 + 86400 * 365)},
            {"user_id": "u", "product_id": "leak", "split": "test", "label": "1", "event_weight": "100", "timestamp": str(1000 + 86400 * 400)},
        ]
        profile = UserProfileBuilder(half_life_days=30).build("u", interactions, products)
        self.assertEqual(profile.preferred_categories[0], "Beauty")
        self.assertNotIn("Automotive", profile.preferred_categories)
        self.assertEqual(profile.interaction_count, 2)

    def test_missing_user_returns_empty_profile(self):
        profile = UserProfileBuilder().build("missing", [], [])
        self.assertEqual(profile.user_id, "missing")
        self.assertEqual(profile.preferred_categories, [])

    def test_review_terms_only_use_products_present_in_train(self):
        products = [
            {"product_id": "train", "category": "Electronics"},
            {"product_id": "test", "category": "Beauty"},
        ]
        interactions = [
            {"user_id": "u", "product_id": "train", "split": "train", "label": "1", "timestamp": "1"},
            {"user_id": "u", "product_id": "test", "split": "test", "label": "1", "timestamp": "2"},
            {"user_id": "u", "product_id": "train", "split": "test", "label": "1", "timestamp": "3"},
        ]
        reviews = [
            {"user_id": "u", "product_id": "train", "rating": "5", "timestamp": "1", "review_text": "comfortable comfortable"},
            {"user_id": "u", "product_id": "test", "rating": "5", "timestamp": "2", "review_text": "forbiddenleak"},
            {"user_id": "u", "product_id": "train", "rating": "5", "timestamp": "3", "review_text": "sameproductleak"},
        ]
        profile = UserProfileBuilder().build("u", interactions, products, reviews)
        self.assertIn("review_term:comfortable", profile.positive_preferences)
        self.assertNotIn("review_term:forbiddenleak", profile.positive_preferences)
        self.assertNotIn("review_term:sameproductleak", profile.positive_preferences)

    def test_review_join_preserves_millisecond_timestamp_precision(self):
        products = [{"product_id": "same", "category": "Electronics"}]
        interactions = [
            {"user_id": "u", "product_id": "same", "split": "train", "label": "1", "timestamp": "1700000000001"},
            {"user_id": "u", "product_id": "same", "split": "test", "label": "1", "timestamp": "1700000000002"},
        ]
        reviews = [
            {"user_id": "u", "product_id": "same", "rating": "5", "timestamp": "1700000000001", "review_text": "traininggoodword"},
            {"user_id": "u", "product_id": "same", "rating": "5", "timestamp": "1700000000002", "review_text": "heldoutbadword"},
        ]
        profile = UserProfileBuilder(top_n=10).build("u", interactions, products, reviews)
        self.assertIn("review_term:traininggoodword", profile.positive_preferences)
        self.assertNotIn("review_term:heldoutbadword", profile.positive_preferences)


if __name__ == "__main__":
    unittest.main()
