import json
import tempfile
import unittest
from pathlib import Path

from em_skillopt.core.types import EpisodeResult, EpisodeSpec
from em_skillopt.skillopt.rollout import RolloutConfig, rollout_specs


class _FakeEnv:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeRunner:
    def __init__(self, capability, calls):
        self.capability = capability
        self.calls = calls
        self.env = _FakeEnv()

    def run_episode(self, spec, skill=None):
        self.calls.append((spec.episode_id, skill))
        return EpisodeResult(spec=spec, success=True, reward=1.0)


class TestRollout(unittest.TestCase):
    def test_rollout_saves_episodes_and_rollout_index(self):
        specs = [
            EpisodeSpec(
                "eb-nav:base:0001", "eb-nav", 1, "go",
                capability="base", split="train",
            ),
            EpisodeSpec(
                "eb-nav:common_sense:0001", "eb-nav", 1, "go",
                capability="common_sense", split="train",
            ),
        ]
        calls = []
        runners = []

        def factory(capability):
            runner = _FakeRunner(capability, calls)
            runners.append(runner)
            return runner

        with tempfile.TemporaryDirectory() as tmp:
            results = rollout_specs(
                specs,
                "learned skill",
                out_dir=tmp,
                runner_factory=factory,
                config=RolloutConfig(verbose=False),
            )
            root = Path(tmp)
            self.assertTrue((root / "rollout_config.json").exists())
            self.assertTrue((root / "summary.json").exists())
            rows = json.loads((root / "rollouts.json").read_text())["rollouts"]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(call[1] == "learned skill" for call in calls))
            self.assertTrue(all(r.env.closed for r in runners))
            for row in rows:
                self.assertTrue(Path(row["episode_json"]).exists())

        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
