import unittest

from src.personalization.recommender import PersonalizedRecommender


def product(
    product_id: str,
    category: str,
    rating: float,
    title: str,
    image_url: str = "",
) -> dict:
    return {
        "product_id": product_id,
        "title": title,
        "category": category,
        "store": "demo-store",
        "price": 25.0,
        "average_rating": rating,
        "rating_number": 20,
        "description": title,
        "features": [],
        "image_url": image_url,
        "preferred_price_range": "mid_range",
    }


class NeedRecommenderTests(unittest.TestCase):
    def setUp(self) -> None:
        products = [
            product("electronic-good", "Electronics", 4.9, "Useful wireless headphones"),
            product("electronic-low", "Electronics", 2.0, "Basic cable"),
            product("beauty", "Beauty_and_Personal_Care", 5.0, "Skin care cream"),
        ]
        embeddings = {
            "electronic-good": [1.0, 0.0],
            "electronic-low": [0.9, 0.1],
            "beauty": [0.0, 1.0],
        }
        modalities = {
            product_id: {
                "text": vector,
                "image": vector,
                "metadata": vector,
                "fused": vector,
            }
            for product_id, vector in embeddings.items()
        }
        self.recommender = PersonalizedRecommender(
            users=[],
            products=products,
            interactions=[],
            product_embeddings=embeddings,
            modality_embeddings=modalities,
        )

    def test_vietnamese_need_filters_category_and_returns_explanation(self) -> None:
        rows = self.recommender.recommend_for_need(
            "Tôi cần tai nghe điện tử được đánh giá tốt",
            top_k=5,
        )

        self.assertEqual([row["category"] for row in rows], ["Electronics", "Electronics"])
        self.assertEqual(rows[0]["product_id"], "electronic-good")
        self.assertIn("match_percentage", rows[0])
        self.assertIn("Nhu cầu", rows[0]["score_breakdown"])
        self.assertIn("Được chọn vì", rows[0]["explanation"])

    def test_need_trace_records_pipeline_counts_and_top_results(self) -> None:
        trace = {}

        rows = self.recommender.recommend_for_need(
            "Tôi cần tai nghe điện tử",
            top_k=2,
            trace=trace,
        )

        self.assertEqual(trace["status"], "ok")
        self.assertEqual(trace["candidate_count_before_seen_filter"], 2)
        self.assertEqual(trace["returned_count"], len(rows))
        self.assertEqual(trace["seen_item_leakage_count"], 0)
        self.assertEqual(trace["top_results"][0]["product_id"], rows[0]["product_id"])
        self.assertIn("headphone", trace["need_tokens"])
        self.assertGreater(trace["semantic_match_result_count"], 0)
        self.assertGreaterEqual(trace["total_ms"], 0.0)

    def test_vietnamese_d_stroke_is_normalized(self) -> None:
        self.assertEqual(
            self.recommender.infer_need_categories("Tôi cần đồ điện tử"),
            ["Electronics"],
        )

    def test_cold_start_personalization_is_neutral_not_popularity(self) -> None:
        self.assertEqual(
            self.recommender._personalization_score("", "electronic-good"),
            0.5,
        )

    def test_similar_products_stay_in_functional_category(self) -> None:
        rows = self.recommender.similar_products("electronic-good", top_k=5)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["product_id"], "electronic-low")
        self.assertIn("Image signal", rows[0]["score_breakdown"])

    def test_need_results_prefer_items_with_images_for_demo(self) -> None:
        self.recommender.products["electronic-low"]["image_url"] = "https://example.com/item.jpg"

        rows = self.recommender.recommend_for_need("electronic product", top_k=1)

        self.assertEqual(rows[0]["product_id"], "electronic-low")

    def test_can_search_a_different_need_on_the_same_recommender_session(self) -> None:
        electronics = self.recommender.recommend_for_need("Tôi cần tai nghe điện tử", top_k=1)
        beauty = self.recommender.recommend_for_need("Tôi cần sản phẩm chăm sóc da", top_k=1)

        self.assertEqual(electronics[0]["category"], "Electronics")
        self.assertEqual(beauty[0]["category"], "Beauty_and_Personal_Care")

    def test_unsupported_appliance_need_does_not_return_unrelated_products(self) -> None:
        rows = self.recommender.recommend_for_need("Tôi cần một tủ lạnh", top_k=5)

        self.assertEqual(self.recommender.infer_need_categories("Tôi cần một tủ lạnh"), ["Appliances"])
        self.assertEqual(rows, [])

    def test_tshirt_need_uses_broad_amazon_fashion_category(self) -> None:
        fashion = PersonalizedRecommender(
            users=[],
            products=[
                product("tee", "Amazon_Fashion", 4.8, "Comfortable cotton t-shirt"),
                product("dress", "Amazon_Fashion", 5.0, "Elegant evening dress"),
            ],
            interactions=[],
            product_embeddings={},
        )

        rows = fashion.recommend_for_need("Tôi cần tìm áo thun", top_k=1)

        self.assertEqual(rows[0]["product_id"], "tee")
        self.assertEqual(fashion.resolve_need_categories("áo thun"), (["Amazon_Fashion"], []))


if __name__ == "__main__":
    unittest.main()
