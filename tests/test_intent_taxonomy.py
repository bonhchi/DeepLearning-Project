import unittest

from src.nlp.intent_taxonomy import INTENT_TAXONOMY, get_intent_definition


class IntentTaxonomyTests(unittest.TestCase):
    def test_taxonomy_has_six_documented_intents(self):
        self.assertEqual(len(INTENT_TAXONOMY), 6)
        for definition in INTENT_TAXONOMY.values():
            self.assertTrue(definition.vietnamese_examples)
            self.assertTrue(definition.english_examples)
            self.assertTrue(definition.distinction_rule)

    def test_unknown_label_is_rejected(self):
        with self.assertRaises(ValueError):
            get_intent_definition("unknown")


if __name__ == "__main__":
    unittest.main()
