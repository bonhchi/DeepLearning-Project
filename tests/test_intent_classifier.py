import tempfile
import unittest
from pathlib import Path

from src.nlp.intent_classifier import IntentClassifier, IntentClassifierNotTrainedError


TRAINING_ROWS = [
    {"query_text": "find wireless headphones", "intent": "product_search"},
    {"query_text": "tìm tai nghe bluetooth", "intent": "product_search"},
    {"query_text": "show me running shoes", "intent": "product_search"},
    {"query_text": "compare these two headphones", "intent": "comparison"},
    {"query_text": "so sánh hai tai nghe này", "intent": "comparison"},
    {"query_text": "compare shoes with other models", "intent": "comparison"},
]


class IntentClassifierTests(unittest.TestCase):
    def test_python_backend_predicts_with_confidence_and_probabilities(self) -> None:
        classifier = IntentClassifier(backend="python", max_iter=120, learning_rate=0.3)
        classifier.train(TRAINING_ROWS)

        prediction = classifier.predict("compare two headphone models")
        probabilities = classifier.predict_proba("compare two headphone models")

        self.assertEqual(prediction["intent"], "comparison")
        self.assertGreater(float(prediction["confidence"]), 0.5)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)
        self.assertEqual(len(classifier.predict_batch(["find shoes", "so sánh giày"])), 2)

    def test_round_trip_persistence_does_not_require_sklearn(self) -> None:
        classifier = IntentClassifier(backend="python", max_iter=80).train(TRAINING_ROWS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intent.pkl"
            classifier.save(path)
            restored = IntentClassifier.load(path)
            self.assertEqual(
                classifier.predict("find headphones"),
                restored.predict("find headphones"),
            )

    def test_untrained_and_empty_queries_fail_clearly(self) -> None:
        classifier = IntentClassifier(backend="python")
        with self.assertRaises(IntentClassifierNotTrainedError):
            classifier.predict("find shoes")
        classifier.train(TRAINING_ROWS)
        with self.assertRaises(ValueError):
            classifier.predict("   ")


if __name__ == "__main__":
    unittest.main()

