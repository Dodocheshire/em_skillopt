"""Domain prompt builders for the four EM environments.

This module deliberately imports EmbodiedBench lazily inside each builder, so the
core package can still be imported in a lightweight Python environment.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..core.types import EpisodeSpec, Message, Observation, TrajectoryStep
from .context import ContextBuilder, format_history


def _action_text(actions: Sequence[Any]) -> str:
    parts = []
    for index, name in enumerate(actions):
        parts.append(f"\naction id {index}: {name}")
        if index < len(actions) - 1:
            parts.append(", ")
    return "".join(parts)


def _default_n_shots(env_name: str) -> int:
    return {
        "eb-nav": 3,
        "eb-alf": 10,
        "eb-hab": 10,
        "eb-man": 4,
    }.get(env_name, 3)


def _load_json_examples(filename: str) -> list[str]:
    import embodiedbench

    package_root = Path(embodiedbench.__path__[0])
    path = package_root / "evaluator" / "config" / filename
    return [str(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def _examples(env_name: str, eval_set: str, n_shots: int, observation: Observation) -> list[str]:
    if env_name == "eb-nav":
        from embodiedbench.evaluator.config.eb_navigation_example import examples
        return [str(item) for item in examples[:n_shots]]

    if env_name == "eb-alf":
        filename = (
            "alfred_long_horizon_examples.json"
            if eval_set == "long_horizon"
            else "alfred_examples.json"
        )
        return _load_json_examples(filename)[:n_shots]

    if env_name == "eb-hab":
        return _load_json_examples("habitat_examples.json")[:n_shots]

    if env_name == "eb-man":
        from embodiedbench.evaluator.config.eb_manipulation_example import (
            llm_examples,
            vlm_examples_baseline,
        )

        variation = str(
            observation.metadata.get("task_variation")
            or getattr(observation, "metadata", {}).get("task_variation")
            or ""
        )
        task_class = variation.split("_")[0] if variation else "pick"
        table = llm_examples if variation and not observation.images else vlm_examples_baseline
        values = table.get(task_class) or table.get("pick") or []
        return [str(item) for item in values[:n_shots]]

    raise ValueError(f"unknown env: {env_name}")


def _output_template(env_name: str, language_only: bool) -> str:
    from embodiedbench.planner.planner_utils import (
        template,
        template_lang,
        template_lang_manip,
        template_manip,
    )

    if env_name == "eb-man":
        return template_lang_manip if language_only else template_manip
    return template_lang if language_only else template


def _system_prompt(
    env_name: str,
    env: Any,
    eval_set: str,
    n_shots: int,
    observation: Observation,
    language_only: bool,
    planner_step: int = 0,
) -> str:
    actions = list(getattr(env, "language_skill_set", []) or [])
    action_text = _action_text(actions)
    raw_examples = _examples(env_name, eval_set, n_shots, observation)
    example_text = "\n\n".join(
        (
            (
                f"## Task Execution Example{'  ' if planner_step > 0 else ' '}{index}: \n {example}"
            )
            if env_name == "eb-nav"
            else example
        )
        for index, example in enumerate(raw_examples)
    )

    if env_name == "eb-nav":
        from embodiedbench.evaluator.config.system_prompts import (
            eb_navigation_system_prompt,
        )
        return eb_navigation_system_prompt.format(
            len(actions) - 1,
            action_text,
            example_text,
        )

    if env_name == "eb-alf":
        from embodiedbench.evaluator.config.system_prompts import (
            alfred_system_prompt,
        )
        return alfred_system_prompt.format(
            len(actions) - 1,
            action_text,
            example_text,
        )

    if env_name == "eb-hab":
        from embodiedbench.evaluator.config.system_prompts import (
            habitat_system_prompt,
        )
        return habitat_system_prompt.format(
            len(actions) - 1,
            action_text,
            example_text,
        )

    if env_name == "eb-man":
        from embodiedbench.envs.eb_manipulation.eb_man_utils import (
            ROTATION_RESOLUTION,
            VOXEL_SIZE,
        )
        from embodiedbench.evaluator.config.system_prompts import (
            eb_manipulation_system_prompt,
        )
        return eb_manipulation_system_prompt.format(
            VOXEL_SIZE,
            VOXEL_SIZE,
            int(360 / ROTATION_RESOLUTION),
            ROTATION_RESOLUTION,
            example_text,
        )

    raise ValueError(f"unknown env: {env_name}")


def _nav_first_prompt(actions: Sequence[Any], template: str) -> str:
    return f"""To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided \
\nAim for about 1-2 actions in this step. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(actions) - 1}) from the available actions to execute. 

The input given to you is an first person view observation . Plan accordingly based on the visual observation.

You are supposed to output in JSON.{template}"""


def _nav_following_prompt(actions: Sequence[Any], template: str) -> str:
    return f"""To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided \
\nAim for about 5-6 actions in this step to be closer to the target object. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(actions) - 1}) from the available actions to execute. 

The input given to you is an first person view observation . Plan accordingly based on the visual observation.

You are supposed to output in JSON.{template}"""


def _nav_action_name(actions: Sequence[Any], action: Any) -> str:
    if isinstance(action, int) and 0 <= action < len(actions):
        return str(actions[action])
    return str(action)


def _make_user_builder(
    env_name: str,
    language_only: bool,
    env: Any,
):
    output_template = _output_template(env_name, language_only)
    actions = list(getattr(env, "language_skill_set", []) or [])

    if env_name == "eb-nav":
        def build_nav_user(
            spec: EpisodeSpec,
            observation: Observation,
            history: Sequence[TrajectoryStep],
            planner_step: int,
        ) -> str:
            instruction = (
                observation.text.strip() or spec.instruction
            ).rstrip(".")
            if planner_step == 0:
                return (
                    f"## Now the human instruction is: {instruction}."
                    + _nav_first_prompt(actions, output_template)
                )

            parts = [
                f"## Now the human instruction is: {instruction}.",
                "\n The action history:",
            ]
            for index, step in enumerate(history):
                parts.append(
                    " Step {}, action id {}, {}, env feedback: {}".format(
                        index,
                        step.action,
                        _nav_action_name(actions, step.action),
                        step.feedback,
                    )
                )
            parts.append("")
            parts.append(_nav_following_prompt(actions, output_template))
            return "\n".join(parts)

        return build_nav_user

    def build_user(
        spec: EpisodeSpec,
        observation: Observation,
        history: Sequence[TrajectoryStep],
        planner_step: int,
    ) -> str:
        instruction = observation.text.strip() or spec.instruction
        parts = [f"## Instruction\n{instruction}"]

        if env_name == "eb-man":
            coords = observation.metadata.get("object_coords", {})
            parts.append(f"## Input object coordinates\n{coords}")

        history_text = format_history(history)
        if history_text:
            parts.append(history_text)

        if planner_step == 0:
            parts.append(
                "This is the first planning step. Describe the current state "
                "and output an initial executable plan."
            )
        else:
            parts.append(
                "The previous plan did not finish the task. Use the interaction "
                "history and environment feedback to explain what failed, then "
                "output a revised plan."
            )

        parts.append(output_template.strip())
        return "\n\n".join(parts)

    return build_user


@dataclass
class DomainContext:
    """ContextBuilder-compatible object built from the live EM environment."""

    env_name: str
    env: Any
    eval_set: str = "base"
    n_shots: int = -1
    language_only: bool = False

    def build(
        self,
        spec: EpisodeSpec,
        observation: Observation,
        history: Sequence[TrajectoryStep],
        skill: str | None,
        *,
        planner_step: int,
    ) -> list[Message]:
        shots = (
            self.n_shots
            if self.n_shots >= 0
            else _default_n_shots(self.env_name)
        )
        base = _system_prompt(
            self.env_name,
            self.env,
            self.eval_set,
            shots,
            observation,
            self.language_only,
            planner_step=planner_step,
        )
        builder = ContextBuilder(
            base_system_prompt=base,
            user_builder=_make_user_builder(
                self.env_name,
                self.language_only,
                self.env,
            ),
            image_first=self.env_name in {"eb-nav", "eb-alf", "eb-hab"},
            system_as_user=True,
        )
        return builder.build(
            spec,
            observation,
            history,
            skill,
            planner_step=planner_step,
        )




def response_format_for(env_name: str, *, language_only: bool = False) -> dict[str, Any]:
    """Return the EM JSON-schema response format for one domain."""

    if env_name == "eb-man":
        from embodiedbench.planner.planner_config.generation_guide_manip import (
            llm_generation_guide_manip,
            vlm_generation_guide_manip,
        )
        schema = llm_generation_guide_manip if language_only else vlm_generation_guide_manip
    else:
        from embodiedbench.planner.planner_config.generation_guide import (
            llm_generation_guide,
            vlm_generation_guide,
        )
        schema = llm_generation_guide if language_only else vlm_generation_guide
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "embodied_planning",
            "schema": schema,
        },
    }


def build_context(
    env_name: str,
    env: Any,
    *,
    eval_set: str = "base",
    n_shots: int = -1,
    language_only: bool = False,
) -> DomainContext:
    return DomainContext(
        env_name=env_name,
        env=env,
        eval_set=eval_set,
        n_shots=n_shots,
        language_only=language_only,
    )


__all__ = ["DomainContext", "build_context", "response_format_for"]
