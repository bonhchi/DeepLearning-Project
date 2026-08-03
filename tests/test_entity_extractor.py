import unittest

from src.nlp.entity_extractor import ENTITY_FIELDS, EntityExtractor, extract_entities


class EntityExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = EntityExtractor()

    def test_empty_query_returns_complete_empty_schema(self) -> None:
        entities = self.extractor.extract("")

        self.assertEqual(tuple(entities), ENTITY_FIELDS)
        self.assertIsNone(entities["category"])
        self.assertEqual(entities["feature"], [])
        self.assertEqual(entities["purpose"], [])
        self.assertIsNone(entities["min_price"])
        self.assertIsNone(entities["max_price"])

    def test_unknown_query_does_not_create_entities(self) -> None:
        entities = self.extractor.extract("Tôi đã tìm sản phẩm đó")

        self.assertTrue(
            all(
                value in (None, [])
                for value in entities.values()
            )
        )

    def test_extracts_vietnamese_entities_with_confidence(self) -> None:
        entities = self.extractor.extract(
            "Tôi cần tai nghe Sony màu đen chống ồn để đi du lịch"
        )

        self.assertEqual(entities["category"]["value"], "Electronics")
        self.assertEqual(entities["brand"]["value"], "Sony")
        self.assertEqual(entities["color"]["value"], "black")
        self.assertEqual(entities["feature"][0]["value"], "noise_cancelling")
        self.assertEqual(entities["purpose"][0]["value"], "travel")
        self.assertGreaterEqual(entities["category"]["confidence"], 0.8)
        self.assertEqual(entities["brand"]["matched_text"], "Sony")

    def test_extracts_english_entities(self) -> None:
        entities = self.extractor.extract(
            "black Bose wireless headphones for gaming"
        )

        self.assertEqual(entities["category"]["value"], "Electronics")
        self.assertEqual(entities["brand"]["value"], "Bose")
        self.assertEqual(entities["color"]["value"], "black")
        self.assertEqual(entities["feature"][0]["value"], "wireless")
        self.assertEqual(entities["purpose"][0]["value"], "gaming")

    def test_extracts_vietnamese_upper_price_constraint(self) -> None:
        entities = self.extractor.extract("Tôi cần laptop dưới 2 triệu")

        self.assertIsNone(entities["min_price"])
        self.assertEqual(entities["max_price"]["value"], 2_000_000)
        self.assertEqual(entities["max_price"]["currency"], "VND")
        self.assertGreaterEqual(entities["max_price"]["confidence"], 0.95)

    def test_extracts_vietnamese_price_range(self) -> None:
        entities = self.extractor.extract("giày từ 500k đến 1 triệu")

        self.assertEqual(entities["min_price"]["value"], 500_000)
        self.assertEqual(entities["max_price"]["value"], 1_000_000)
        self.assertEqual(entities["min_price"]["currency"], "VND")
        self.assertEqual(entities["max_price"]["currency"], "VND")

    def test_extracts_english_dollar_upper_price_constraint(self) -> None:
        entities = extract_entities("wireless headphones less than $100")

        self.assertEqual(entities["max_price"]["value"], 100)
        self.assertEqual(entities["max_price"]["currency"], "USD")

    def test_extracts_grouped_exact_dollar_amount_without_truncation(self) -> None:
        entities = extract_entities("camera price $1,299.99")

        self.assertEqual(entities["min_price"]["value"], 1299.99)
        self.assertEqual(entities["max_price"]["value"], 1299.99)
        self.assertEqual(entities["max_price"]["matched_text"], "$1,299.99")
        self.assertEqual(entities["max_price"]["currency"], "USD")

    def test_plain_numeric_ranges_are_not_misclassified_as_prices(self) -> None:
        for query in (
            "14 and 15",
            "from 14 to 15",
            "size 8 to 10",
            "model 100 to 200",
        ):
            with self.subTest(query=query):
                entities = extract_entities(query)
                self.assertIsNone(entities["min_price"])
                self.assertIsNone(entities["max_price"])

    def test_product_model_suffix_d_is_not_a_dong_price(self) -> None:
        for query in ("Canon 5D Mark IV", "camera model 5d"):
            with self.subTest(query=query):
                entities = extract_entities(query)
                self.assertIsNone(entities["min_price"])
                self.assertIsNone(entities["max_price"])

        contextual = extract_entities("camera price 500d")
        self.assertEqual(contextual["min_price"]["value"], 500)
        self.assertEqual(contextual["max_price"]["currency"], "VND")

    def test_range_with_price_or_currency_context_is_extracted(self) -> None:
        by_label = extract_entities("headphones price from 100 to 200")
        by_currency = extract_entities("headphones from $100 to $200")
        after_size = extract_entities("size 8 to 10, price from 100 to 200 USD")

        self.assertEqual(by_label["min_price"]["value"], 100)
        self.assertEqual(by_label["max_price"]["value"], 200)
        self.assertEqual(by_currency["min_price"]["currency"], "USD")
        self.assertEqual(by_currency["max_price"]["currency"], "USD")
        self.assertEqual(after_size["min_price"]["value"], 100)
        self.assertEqual(after_size["max_price"]["value"], 200)

    def test_scale_and_explicit_currency_are_parsed_from_same_amount(self) -> None:
        entities = extract_entities("price 2 million VND")

        self.assertEqual(entities["min_price"]["value"], 2_000_000)
        self.assertEqual(entities["max_price"]["currency"], "VND")
        self.assertEqual(entities["max_price"]["matched_text"], "2 million VND")

    def test_mixed_currency_range_keeps_currency_per_bound(self) -> None:
        entities = extract_entities("100 USD to 2 million VND")

        self.assertEqual(entities["min_price"]["value"], 100)
        self.assertEqual(entities["min_price"]["currency"], "USD")
        self.assertEqual(entities["max_price"]["value"], 2_000_000)
        self.assertEqual(entities["max_price"]["currency"], "VND")

    def test_reversed_same_currency_range_is_normalized(self) -> None:
        entities = extract_entities("price between 40 and 20")

        self.assertEqual(entities["min_price"]["value"], 20)
        self.assertEqual(entities["max_price"]["value"], 40)

    def test_extracts_explicit_size_and_material(self) -> None:
        entities = self.extractor.extract("áo cotton màu đỏ size XL")

        self.assertEqual(entities["size"]["value"], "XL")
        self.assertEqual(entities["material"]["value"], "cotton")
        self.assertEqual(entities["color"]["value"], "red")

    def test_marks_an_explicitly_negated_feature(self) -> None:
        entities = self.extractor.extract("headphones not waterproof")

        self.assertEqual(entities["feature"][0]["value"], "waterproof")
        self.assertTrue(entities["feature"][0]["negated"])


if __name__ == "__main__":
    unittest.main()
