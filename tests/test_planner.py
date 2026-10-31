import unittest
from types import SimpleNamespace

from src.llm_planner import LLMPlanner


class FakeModels:
    def __init__(self, text):
        self.text = text

    def generate_content(self, **_kwargs):
        return SimpleNamespace(text=self.text)


class FakeClient:
    def __init__(self, text):
        self.models = FakeModels(text)


class PlannerTests(unittest.TestCase):
    def test_idea_validation_rejects_unsupported_parameters(self):
        invalid = LLMPlanner._validate_idea(
            {
                "model": "Random Forest",
                "hyperparameters": {"n_estimators": 9999},
            },
            "HIV",
        )
        self.assertIsNone(invalid)

    def test_idea_validation_sets_task_specific_features(self):
        idea = LLMPlanner._validate_idea(
            {"model": "Random Forest", "hyperparameters": {"n_estimators": 200}},
            "DAVIS",
        )
        self.assertEqual(idea["featurization"], "ECFP4_plus_CT")
        self.assertEqual(idea["hyperparameters"]["n_estimators"], 200)

    def test_generate_ideas_parses_and_validates_json(self):
        payload = '[{"model":"Logistic Regression","hyperparameters":{"C":0.1,"class_weight":"balanced"},"rationale":"Baseline"}]'
        planner = LLMPlanner(client=FakeClient(payload))
        ideas = planner.generate_ideas("HIV", k=1)
        self.assertEqual(len(ideas), 1)
        self.assertEqual(ideas[0]["model"], "Logistic Regression")
        self.assertEqual(ideas[0]["featurization"], "ECFP4_Morgan")

    def test_revise_ideas_uses_best_validation_score(self):
        planner = LLMPlanner(client=FakeClient('{"reasoning":"The first candidate has the best validation score."}'))
        explored = [
            {"idea": {"model": "Random Forest", "hyperparameters": {"n_estimators": 200}}, "report": {"success": True, "validation_roc_auc": 0.81}},
            {"idea": {"model": "Logistic Regression", "hyperparameters": {"C": 1.0}}, "report": {"success": True, "validation_roc_auc": 0.86}},
        ]
        decision = planner.revise_ideas(explored)
        self.assertEqual(decision["chosen"], explored[1])


if __name__ == "__main__":
    unittest.main()
