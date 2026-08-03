import unittest

from src.app.streamlit_app import display_image_url, execute_search_option, next_result_page


class _FakeSearch:
    last_trace = {"mode": "lexical", "intent": "unknown"}

    def search(self, query, top_k, mode):
        self.last_trace = {"query": query, "mode": mode, "intent": "unknown"}
        return [
            {
                "product_id": mode,
                "title": mode,
                "final_score": 0.8,
                "semantic_score": 0.7,
                "lexical_score": 0.9,
            }
        ]


class _FakeRouter:
    def route(self, query, **kwargs):
        return {
            "original_query": query,
            "detected_intent": "product_search",
            "intent_confidence": 0.9,
            "rewritten_query": query,
            "search_mode": kwargs.get("mode"),
            "results": [{"product_id": "aware", "match_percentage": 90.0}],
        }


class StreamlitAppHelpersTests(unittest.TestCase):
    def test_resizes_full_amazon_image(self) -> None:
        url = "https://m.media-amazon.com/images/I/example.jpg"

        self.assertEqual(
            display_image_url(url),
            "https://m.media-amazon.com/images/I/example._SL360_.jpg",
        )

    def test_replaces_existing_amazon_size_modifier(self) -> None:
        url = "https://m.media-amazon.com/images/I/example._AC_SL1600_.jpg"

        self.assertEqual(
            display_image_url(url),
            "https://m.media-amazon.com/images/I/example._SL360_.jpg",
        )

    def test_keeps_non_amazon_url(self) -> None:
        url = "https://example.com/image.jpg"

        self.assertEqual(display_image_url(url), url)

    def test_repeated_search_advances_to_next_result_page(self) -> None:
        key = ("tai nghe", "user-1")

        self.assertEqual(next_result_page(None, key, 3), 0)
        self.assertEqual(next_result_page(key, key, 0), 1)

    def test_ui_can_compare_all_three_search_modes(self) -> None:
        rows, trace, comparison = execute_search_option(
            {"baseline": _FakeSearch(), "router": _FakeRouter()},
            recommender=None,
            query="headphones",
            option="So sánh cả 3 chế độ",
            top_k=5,
        )
        self.assertEqual(rows[0]["product_id"], "aware")
        self.assertEqual(trace["detected_intent"], "product_search")
        self.assertEqual(len(comparison), 3)


if __name__ == "__main__":
    unittest.main()
