import tempfile
import unittest
from pathlib import Path

from em_skillopt.core.artifacts import RunArtifacts
from em_skillopt.core.types import EpisodeResult, EpisodeSpec
from em_skillopt.harness.batch import BatchConfig
from em_skillopt.harness.capability import CapabilityBatchRunner


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
        self.calls.append((self.capability, spec.episode_id, skill))
        return EpisodeResult(spec=spec, success=True, reward=1.0)


class TestCapabilityBatchRunner(unittest.TestCase):
    def test_groups_by_capability_and_closes_each_runner(self):
        specs = [
            EpisodeSpec("eb-nav:common_sense:0001", "eb-nav", 1, "a", capability="common_sense"),
            EpisodeSpec("eb-nav:base:0001", "eb-nav", 1, "b", capability="base"),
            EpisodeSpec("eb-nav:common_sense:0002", "eb-nav", 2, "c", capability="common_sense"),
        ]
        calls = []
        runners = []

        def factory(capability):
            runner = _FakeRunner(capability, calls)
            runners.append(runner)
            return runner

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = RunArtifacts(Path(tmp))
            results = CapabilityBatchRunner(
                factory,
                artifacts=artifacts,
                config=BatchConfig(resume=False),
            ).run(specs, skill="learned")

        self.assertEqual([r.capability for r in runners], ["base", "common_sense"])
        self.assertTrue(all(r.env.closed for r in runners))
        self.assertEqual(
            [call[0] for call in calls],
            ["base", "common_sense", "common_sense"],
        )
        self.assertTrue(all(call[2] == "learned" for call in calls))
        self.assertEqual(len(results), 3)

    def test_resume_skips_done_episode_but_keeps_runner_lifecycle(self):
        specs = [
            EpisodeSpec("eb-nav:base:0001", "eb-nav", 1, "a", capability="base"),
        ]
        calls = []
        runners = []

        def factory(capability):
            runner = _FakeRunner(capability, calls)
            runners.append(runner)
            return runner

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = RunArtifacts(Path(tmp))
            artifacts.save_episode(
                EpisodeResult(spec=specs[0], success=True)
            )
            results = CapabilityBatchRunner(
                factory,
                artifacts=artifacts,
                config=BatchConfig(resume=True),
            ).run(specs)

        self.assertEqual(results, [])
        self.assertEqual(calls, [])
        self.assertEqual(len(runners), 1)
        self.assertTrue(runners[0].env.closed)


if __name__ == "__main__":
    unittest.main()
