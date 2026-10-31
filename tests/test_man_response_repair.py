import json
import unittest
from types import SimpleNamespace

from em_skillopt.core.types import LLMResponse, TokenUsage
from em_skillopt.harness.man_response import request_man_plan
from em_skillopt.harness.parsing import ManipulationPlanParser


class FakeLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def generate(self, messages, **kwargs):
        import copy
        self.requests.append((copy.deepcopy(messages), kwargs))
        return next(self.replies)


def reply(document, finish="stop"):
    return LLMResponse(json.dumps(document), usage=TokenUsage(calls=1), finish_reason=finish)


class ManRepairTests(unittest.TestCase):
    def run_request(self, llm):
        return request_man_plan(llm, [{"role":"user", "content":"task"}],
            ManipulationPlanParser(), max_tokens=8192, temperature=0,
            response_format={"type":"json_schema"})

    def test_repair_contains_bad_output_and_error_then_executes_only_valid_plan(self):
        good = {"executable_plan":[{"action":[50,42,17,6,61,36,1]}]}
        llm = FakeLLM([reply({"executable_plan":[{"action":"move_to()"}]}), reply(good)])
        out = self.run_request(llm)
        self.assertEqual(out.plan.actions, [[50,42,17,6,61,36,1]])
        self.assertEqual(out.usage.calls, 2)
        self.assertEqual(llm.requests[1][0][-2]["role"], "assistant")
        self.assertIn("Your action plan cannot be parsed:", llm.requests[1][0][-1]["content"])

    def test_final_parse_failure_stops_without_executing_actions(self):
        llm = FakeLLM([reply({"executable_plan":"move_to()"}) for _ in range(3)])
        outcome = self.run_request(llm)
        self.assertTrue(outcome.plan.stop)
        self.assertIn("after 3 attempts", outcome.plan.error)
        self.assertEqual(outcome.plan.actions, [])
        self.assertEqual(len(llm.requests), 3)

    def test_final_length_retries_same_planner_once_then_raises(self):
        from unittest.mock import patch
        llm = FakeLLM([reply({}, "length") for _ in range(6)])
        with patch("em_skillopt.harness.man_response.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                self.run_request(llm)
        sleep.assert_called_once_with(60)
        self.assertEqual([kw["max_tokens"] for _,kw in llm.requests],
                         [8192,16384,32768,8192,16384,32768])

    def test_truncation_doubles_token_budget(self):
        llm = FakeLLM([reply({}, "length"), reply({"executable_plan":[]})])
        out = self.run_request(llm)
        self.assertEqual(llm.requests[1][1]["max_tokens"], 16384)
        self.assertTrue(out.plan.stop)

    def test_compatible_wrapped_plan_is_not_rejected_by_schema_first(self):
        llm = FakeLLM([reply({"properties":{"executable_plan":[[50,42,17,6,61,36,1]]}})])
        self.assertEqual(len(self.run_request(llm).plan.actions), 1)


if __name__ == "__main__":
    unittest.main()
