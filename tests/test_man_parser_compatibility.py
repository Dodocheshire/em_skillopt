"""MAN decoder boundary cases from pinned official manip_actions.py semantics."""
import unittest

from em_skillopt.harness.parsing import ManipulationPlanParser


class ManipulationDecoderCompatibility(unittest.TestCase):
    def test_supported_wrappers_and_integral_floats(self):
        action = [50, 42, 17, 6, 61, 36, 1]
        documents = [
            {"executable_plan": [{"action": action}]},
            {"properties": {"executable_plan": [action]}},
            {"executable_plan": str([{"action": action}])},
            {"executable_plan": [[{"action": str(action)}]]},
            {"executable_plan": [[float(x) for x in action]]},
        ]
        for document in documents:
            with self.subTest(document=document):
                result = ManipulationPlanParser().parse_document(document, [])
                self.assertIsNone(result.error)
                self.assertEqual(result.actions, [action])

    def test_invalid_actions_do_not_execute_a_valid_prefix(self):
        valid = [50, 42, 17, 6, 61, 36, 1]
        invalid = [
            [101, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 120, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 2],
            [True, 0, 0, 0, 0, 0, 1],
            [float("nan"), 0, 0, 0, 0, 0, 1],
            [0.5, 0, 0, 0, 0, 0, 1],
        ]
        for action in invalid:
            with self.subTest(action=action):
                result = ManipulationPlanParser().parse_document(
                    {"executable_plan": [valid, action]}, []
                )
                self.assertIsNotNone(result.error)
                self.assertEqual(result.actions, [])

    def test_empty_plan_stops(self):
        result = ManipulationPlanParser().parse_document({"executable_plan": []}, [])
        self.assertTrue(result.stop)
        self.assertIsNone(result.error)


if __name__ == "__main__":
    unittest.main()
