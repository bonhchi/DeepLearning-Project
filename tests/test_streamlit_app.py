import unittest

from src.app.streamlit_app import display_image_url, next_result_page


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


if __name__ == "__main__":
    unittest.main()
