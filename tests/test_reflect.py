import json
import tempfile
import unittest
from pathlib import Path

from em_skillopt.core.types import (
    EpisodeResult,
    EpisodeSpec,
    PlanRecord,
    TrajectoryStep,
)
from em_skillopt.skillopt.reflect import ReflectConfig, episode_digest, reflect_episodes


class _FakeLLM:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        class R:
            pass
        r = R()
        r.text = self.text
        return r


def _result(success=False):
    spec = EpisodeSpec(
        "eb-nav:base:0001", "eb-nav", 1, "go to the bread",
        capability="base", split="train",
    )
    return EpisodeResult(
        spec=spec,
        success=success,
        reward=1.0 if success else 0.0,
        num_env_steps=2,
        num_planner_steps=2,
        invalid_actions=0 if success else 1,
        failure_reason="" if success else "invalid_action",
        plans=[
            PlanRecord(0, '{"executable_plan": [{"action_id": 0}]}', [0]),
            PlanRecord(1, "bad", [], "parse error"),
        ],
        trajectory=[
            TrajectoryStep(0, 1, 0, True, 0.0, "ok", False, False),
            TrajectoryStep(0, 2, 7, False, 0.0, "blocked", False, False),
        ],
    )


class TestReflect(unittest.TestCase):
    def test_digest_is_compact(self):
        d = episode_digest(_result())
        self.assertEqual(d["failure_reason"], "invalid_action")
        self.assertEqual(len(d["plans"]), 2)
        self.assertIn("trajectory_tail", d)

    def test_reflect_saves_digests_and_patch(self):
        llm = _FakeLLM(
            '{"reasoning": "avoid blocked action", "edits": ['
            '{"op": "append", "content": "rotate when blocked", '
            '"source_type": "failure"}]}'
        )
        with tempfile.TemporaryDirectory() as tmp:
            patches = reflect_episodes(
                [_result(False)],
                "current skill",
                llm=llm,
                out_dir=tmp,
                config=ReflectConfig(max_episodes=4),
            )
            self.assertEqual(len(patches), 1)
            self.assertEqual(patches[0]["edits"][0]["content"], "rotate when blocked")
            self.assertTrue((Path(tmp) / "digests.json").exists())
            saved = json.loads((Path(tmp) / "patches.json").read_text())
            self.assertEqual(saved["patches"][0]["edits"][0]["op"], "append")
            self.assertEqual(len(llm.calls), 1)
            self.assertEqual(llm.calls[0][1]["stage"], "reflect")


if __name__ == "__main__":
    unittest.main()
