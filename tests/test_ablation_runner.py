import unittest

from src.evaluation.ablation_runner import AblationRunner


class AblationRunnerTests(unittest.TestCase):
    def test_runs_full_and_component_ablations(self):
        report = AblationRunner(lambda config: {"ndcg@5": int(config.use_intent_detection)}).run()
        self.assertEqual(len(report["results"]), 5)
        names = {row["config"]["name"] for row in report["results"]}
        self.assertIn("without_personalization", names)


if __name__ == "__main__":
    unittest.main()
