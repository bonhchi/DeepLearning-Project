import unittest

from src.personalization.explanation_builder import ExplanationBuilder


class ExplanationBuilderTests(unittest.TestCase):
    def test_mentions_evidence_instead_of_generic_copy(self):
        text = ExplanationBuilder().build(
            product={"title": "Headphones", "price": "80", "category": "Electronics"},
            intent="need_based_search",
            score_breakdown={"semantic_score": 0.9, "quality_score": 0.8},
            matched_entities={"feature": {"value": "noise cancelling"}, "max_price": {"value": 100}},
        )
        self.assertIn("noise cancelling", text)
        self.assertIn("mức giá", text)


if __name__ == "__main__":
    unittest.main()
