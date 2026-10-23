"""Comprehensive smoke/unit tests for the EM-SkillOpt harness.

Run with::

    cd /home/pengcheng/EM-Benchmark/em_skillopt
    PYTHONPATH=. python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from em_skillopt.core.artifacts import RunArtifacts
from em_skillopt.core.metrics import em_summary, summarize
from em_skillopt.core.skill import SkillStore, render_skill_block
from em_skillopt.core.types import (
    ActionPlan,
    EpisodeResult,
    EpisodeSpec,
    LLMResponse,
    Observation,
    StepResult,
    TokenUsage,
)
from em_skillopt.harness.context import (
    ContextBuilder,
    SimpleUserPromptBuilder,
    build_system_prompt,
)
from em_skillopt.harness.parsing import (
    ActionIdPlanParser,
    ManipulationPlanParser,
    load_json_object,
)
from em_skillopt.harness.runner import EpisodeRunner, RunnerConfig
from em_skillopt.harness.adapters.alf import AlfAdapter
from em_skillopt.harness.adapters.hab import HabAdapter
from em_skillopt.harness.adapters.man import ManAdapter
from em_skillopt.harness.adapters.nav import NavAdapter


def make_spec(index: int = 0, max_steps: int = 5) -> EpisodeSpec:
    return EpisodeSpec(
        episode_id=f"test:{index:04d}",
        env_name="test",
        index=index,
        instruction="do the task",
        capability="base",
        split="test",
        max_steps=max_steps,
    )


class FakeContext:
    def build(self, spec, observation, history, skill, *, planner_step):
        return [{"role": "user", "content": "test"}]


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def generate(self, messages, *, max_tokens, temperature=None, response_format=None, stage="target"):
        text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return LLMResponse(
            text=text,
            model="fake",
            usage=TokenUsage(calls=1, prompt_tokens=10, completion_tokens=5),
        )


class RunnerEnv:
    action_space = [0, 1]
    max_steps = 5
    replan_on_invalid = True

    def __init__(self) -> None:
        self.steps = 0

    def reset(self, spec):
        self.steps = 0
        return Observation()

    def step(self, action):
        self.steps += 1
        if action == 0:
            return StepResult(
                observation=Observation(),
                reward=1.0,
                done=True,
                success=True,
                feedback="done",
                action_valid=True,
            )
        return StepResult(
            observation=Observation(),
            reward=0.0,
            done=False,
            success=False,
            feedback="invalid",
            action_valid=False,
        )

    def close(self):
        pass


class TestSkillAndPrompt(unittest.TestCase):
    def test_baseline_prompt_is_unchanged(self):
        base = "BASE PROMPT\n"
        self.assertEqual(build_system_prompt(base, None), base)
        self.assertEqual(build_system_prompt(base, "   "), base)
        self.assertEqual(render_skill_block(None), "")

    def test_method_a_prompt_adds_one_block(self):
        base = "BASE PROMPT"
        prompt = build_system_prompt(base, "- always check the object first")
        self.assertTrue(prompt.startswith(base))
        self.assertIn("## Learned Skill (Method A)", prompt)
        self.assertEqual(prompt.count("## Learned Skill (Method A)"), 1)

    def test_skill_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillStore(tmp)
            saved = store.save("- rule one\n", version=0, promote_best=True)
            self.assertEqual(saved.version, 0)
            self.assertEqual(store.load_current().content.strip(), "- rule one")
            self.assertEqual(store.load_best().content.strip(), "- rule one")
            self.assertEqual(store.list_versions(), [0])

            second = store.save("- rule two\n")
            self.assertEqual(second.version, 1)
            self.assertEqual(store.load_current().content.strip(), "- rule two")
            # best was not promoted, so it remains the first version
            self.assertEqual(store.load_best().content.strip(), "- rule one")
            store.promote_best()
            self.assertEqual(store.load_best().content.strip(), "- rule two")

    def test_context_first_vs_replan(self):
        spec = make_spec()
        observation = Observation(images=["/tmp/frame.png"], text="you see a desk")
        builder = ContextBuilder(
            base_system_prompt="BASE",
            user_builder=SimpleUserPromptBuilder(
                first_step_suffix="FIRST PLAN",
                replan_suffix="REPLAN NOW",
            ),
        )

        first = builder.build(spec, observation, [], None, planner_step=0)
        self.assertEqual(first[0]["content"], "BASE")
        self.assertIn("FIRST PLAN", first[1]["content"][0]["text"])
        self.assertNotIn("REPLAN NOW", first[1]["content"][0]["text"])
        self.assertEqual(first[1]["content"][1]["type"], "image_path")

        replan = builder.build(
            spec,
            observation,
            [],
            "- learned rule",
            planner_step=1,
        )
        self.assertIn("## Learned Skill (Method A)", replan[0]["content"])
        self.assertIn("REPLAN NOW", replan[1]["content"][0]["text"])


class TestParsing(unittest.TestCase):
    def test_json_fence(self):
        raw = '```json\n{"executable_plan": [{"action_id": 1}]}\n```'
        self.assertEqual(load_json_object(raw)["executable_plan"][0]["action_id"], 1)

    def test_high_level_valid(self):
        parser = ActionIdPlanParser()
        plan = parser.parse(
            '{"executable_plan": [{"action_id": 0}, {"action_id": 2}]}',
            ["a", "b", "c"],
        )
        self.assertTrue(plan.valid)
        self.assertEqual(plan.actions, [0, 2])

    def test_high_level_empty_is_stop(self):
        plan = ActionIdPlanParser().parse('{"executable_plan": []}', [0])
        self.assertTrue(plan.stop)
        self.assertFalse(plan.error)

    def test_high_level_invalid_first_has_no_actions(self):
        plan = ActionIdPlanParser().parse(
            '{"executable_plan": [{"action_id": 9}]}',
            [0, 1],
        )
        self.assertFalse(plan.valid)
        self.assertEqual(plan.actions, [])

    def test_high_level_invalid_later_keeps_prefix(self):
        plan = ActionIdPlanParser().parse(
            '{"executable_plan": [{"action_id": 0}, {"action_id": 9}]}',
            [0, 1],
        )
        self.assertFalse(plan.valid)
        self.assertEqual(plan.actions, [0])

    def test_manipulation_valid_and_invalid(self):
        parser = ManipulationPlanParser()
        valid = parser.parse(
            '{"executable_plan": [[1,2,3,4,5,6,1]]}',
            [],
        )
        self.assertTrue(valid.valid)
        self.assertEqual(valid.actions, [[1, 2, 3, 4, 5, 6, 1]])

        bad_range = parser.parse(
            '{"executable_plan": [[101,2,3,4,5,6,1]]}',
            [],
        )
        self.assertFalse(bad_range.valid)
        self.assertEqual(bad_range.actions, [])

        bad_shape = parser.parse(
            '{"executable_plan": [[1,2,3,4,5,6]]}',
            [],
        )
        self.assertFalse(bad_shape.valid)


class TestRunner(unittest.TestCase):
    def _runner(self, responses: list[str]) -> EpisodeRunner:
        return EpisodeRunner(
            env=RunnerEnv(),
            context=FakeContext(),
            parser=ActionIdPlanParser(),
            llm=FakeLLM(responses),
            config=RunnerConfig(max_planner_steps=5, max_steps=5),
        )

    def test_success(self):
        runner = self._runner(['{"executable_plan": [{"action_id": 0}]}'])
        result = runner.run_episode(make_spec())
        self.assertTrue(result.success)
        self.assertEqual(result.num_env_steps, 1)
        self.assertEqual(result.failure_reason, "")

    def test_empty_plan(self):
        runner = self._runner(['{"executable_plan": []}'])
        result = runner.run_episode(make_spec())
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "empty_plan")

    def test_format_error(self):
        runner = self._runner(["not json"])
        result = runner.run_episode(make_spec())
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "format_error")
        self.assertEqual(result.format_errors, 1)

    def test_invalid_action_replans(self):
        runner = self._runner(
            [
                '{"executable_plan": [{"action_id": 1}]}',
                '{"executable_plan": [{"action_id": 0}]}',
            ]
        )
        result = runner.run_episode(make_spec())
        self.assertTrue(result.success)
        self.assertEqual(result.num_planner_steps, 2)
        self.assertEqual(result.invalid_actions, 1)


class FakeNavEnv:
    language_skill_set = ["forward", "back"]
    _max_episode_steps = 20
    _current_episode_num = 0

    def reset(self):
        self._current_episode_num += 1
        return {"head_rgb": None}

    def step(self, action, reasoning, i_flag):
        return {}, 1.0, True, {
            "env_feedback": "arrived",
            "task_success": 1.0,
            "last_action_success": 1.0,
            "env_step": 1,
            "distance": 0.0,
        }

    def save_image(self):
        return "/tmp/nav.png"

    def close(self):
        pass


class FakeAlfEnv:
    language_skill_set = ["find apple", "pick apple"]
    _max_episode_steps = 30
    _current_episode_num = 0
    episode_language_instruction = "pick the apple"

    def reset(self):
        self._current_episode_num += 1
        return {"head_rgb": None}

    def step(self, action, reasoning=""):
        return {}, 1.0, True, {
            "env_feedback": "done",
            "task_success": 1.0,
            "last_action_success": 1.0,
            "env_step": 1,
        }

    def save_image(self):
        return "/tmp/alf.png"

    def close(self):
        pass


class FakeHabEnv:
    language_skill_set = ["navigate", "pick", "place"]
    _max_episode_steps = 30
    _current_episode_num = 0
    episode_language_instruction = "put the cup away"

    def reset(self):
        self._current_episode_num += 1
        return {"head_rgb": None}

    def step(self, action, reasoning=""):
        return {"head_rgb": None}, 1.0, True, {
            "env_feedback": "done",
            "task_success": 1.0,
            "last_action_success": 1.0,
            "env_step": 1,
        }

    def save_image(self, obs):
        return "/tmp/hab.png"

    def close(self):
        pass


class FakeManEnv:
    _max_episode_steps = 15
    _current_episode_num = 0
    task_class = "pick"
    current_task_variation = "pick_cube_shape"

    def __init__(self, description):
        self.description = description

    def reset(self):
        self._current_episode_num += 1
        return self.description, types.SimpleNamespace(object_informations={})

    def step(self, action):
        return types.SimpleNamespace(object_informations={}), 1.0, True, {
            "env_feedback": "done",
            "task_success": 1.0,
            "action_success": 1.0,
            "env_step": 1,
            "episode_num": 1,
        }

    def save_image(self, key=["front_rgb"]):
        return ["/tmp/man.png"]

    def close(self):
        pass


@contextmanager
def fake_man_utils():
    """Provide the tiny part of embodiedbench used by ManAdapter._observe."""

    names = [
        "embodiedbench",
        "embodiedbench.envs",
        "embodiedbench.envs.eb_manipulation",
        "embodiedbench.envs.eb_manipulation.eb_man_utils",
    ]
    modules = {name: types.ModuleType(name) for name in names}

    def form_object_coord_for_input(raw_obs, task_class, camera_views):
        return {"object 1": [1, 2, 3]}, [[1, 2, 3]], [None], [None]

    def draw_xyz_coordinate(path, resolution):
        return path

    def draw_bounding_boxes(images, points, extrinsics, intrinsics):
        return images

    utils = modules[names[-1]]
    utils.form_object_coord_for_input = form_object_coord_for_input
    utils.draw_xyz_coordinate = draw_xyz_coordinate
    utils.draw_bounding_boxes = draw_bounding_boxes

    with patch.dict(sys.modules, modules):
        yield


class TestAdapters(unittest.TestCase):
    def test_nav_adapter(self):
        adapter = NavAdapter(FakeNavEnv())
        observation = adapter.reset(make_spec())
        self.assertEqual(observation.images, ["/tmp/nav.png"])
        step = adapter.step(0)
        self.assertEqual(step.observation.images, ["/tmp/nav.png"])
        self.assertTrue(step.success)
        self.assertTrue(step.done)
        self.assertTrue(step.action_valid)
        self.assertEqual(len(adapter.action_space), 2)

    def test_alf_adapter_updates_action_space(self):
        adapter = AlfAdapter(FakeAlfEnv())
        observation = adapter.reset(make_spec())
        self.assertEqual(observation.text, "pick the apple")
        self.assertEqual(len(adapter.action_space), 2)
        step = adapter.step(0)
        self.assertTrue(step.success)

    def test_hab_adapter_sequential(self):
        env = FakeHabEnv()
        adapter = HabAdapter(env)
        first = adapter.reset(make_spec(index=0))
        self.assertEqual(first.images, ["/tmp/hab.png"])
        second = adapter.reset(make_spec(index=1))
        self.assertEqual(second.images, ["/tmp/hab.png"])
        self.assertEqual(env._current_episode_num, 2)
        step = adapter.step(0)
        self.assertTrue(step.success)

    def test_man_adapter_description_list_and_coords(self):
        with fake_man_utils():
            adapter = ManAdapter(FakeManEnv(["pick the apple"]))
            self.assertEqual(adapter.camera_views, ["front_rgb"])
            observation = adapter.reset(make_spec())
            self.assertEqual(observation.text, "pick the apple")
            self.assertEqual(
                observation.metadata["object_coords"],
                {"object 1": [1, 2, 3]},
            )

            step = adapter.step([1, 2, 3, 4, 5, 6, 1])
            self.assertTrue(step.success)
            self.assertTrue(step.done)
            self.assertTrue(step.action_valid)

    def test_man_adapter_description_string_does_not_take_first_char(self):
        with fake_man_utils():
            adapter = ManAdapter(FakeManEnv("pick the apple"))
            observation = adapter.reset(make_spec())
            self.assertEqual(observation.text, "pick the apple")


class TestArtifactsAndMetrics(unittest.TestCase):
    def test_artifacts_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = RunArtifacts(tmp)
            spec = make_spec()
            result = EpisodeResult(
                spec=spec,
                success=True,
                reward=1.0,
                num_env_steps=3,
                num_planner_steps=1,
                elapsed_s=1.5,
                token_usage=TokenUsage(
                    calls=1,
                    prompt_tokens=10,
                    completion_tokens=2,
                ),
            )
            artifacts.save_episode(result)
            self.assertTrue(artifacts.episode_done(spec.episode_id))
            records = list(artifacts.iter_episode_dicts())
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["success"], True)

            summary = summarize([result], env_name="test", split="test")
            self.assertEqual(summary.total, 1)
            self.assertEqual(summary.success_rate, 1.0)
            em = em_summary([result], env_name="test", split="test")
            self.assertEqual(em["num_success"], 1)
            self.assertEqual(em["output_format_error"], 0)

            artifacts.save_summary(summary)
            self.assertTrue((Path(tmp) / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
