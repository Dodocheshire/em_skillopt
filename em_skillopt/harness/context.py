"""general prompt assembly
[base system prompt]        # 原 EM prompt，baseline/Method A 相同
[learned skill block]       # 仅 Method A；baseline 没有
[action space / skill set]
[examples]
[instruction]
[current observation]
[history + env feedback]
[output JSON schema]

Prompt composition for the target model.

This module owns two things:

1. The only place where a learned skill is injected into the target prompt.
2. Conversion from ``Observation`` + ``TrajectoryStep`` history into chat
   messages.

Domain-specific prompt text (ALFRED / Habitat / Navigation / Manipulation) is
supplied by the domain adapter as ``base_system_prompt`` and ``user_builder``.
The builder here stays domain-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..core.skill import render_skill_block
from ..core.types import (
    EpisodeSpec,
    Message,
    Observation,
    TrajectoryStep,
)

# A domain adapter supplies this callable.
UserPromptBuilder = Callable[
    [EpisodeSpec, Observation, Sequence[TrajectoryStep], int],
    str,
]


def build_system_prompt(base_system_prompt: str, skill: str | None) -> str:
    """Return the target system prompt.

    Baseline (``skill=None`` or empty) must be byte-identical to the original
    EM prompt.  Method A appends exactly one skill block.
    """

    block = render_skill_block(skill)
    if not block:
        return base_system_prompt
    return f"{base_system_prompt.rstrip()}\n\n{block}"


def format_history(
    history: Sequence[TrajectoryStep],
    *,
    max_steps: int = 20,
    feedback_limit: int = 500,
) -> str:
    """Format recent executed actions for replanning prompts."""

    if not history:
        return ""

    lines = ["## Interaction History"]
    for step in list(history)[-max_steps:]:
        feedback = (step.feedback or "").strip()
        if len(feedback) > feedback_limit:
            feedback = feedback[:feedback_limit] + " ..."

        line = (
            f"- env step {step.env_step} "
            f"(planner step {step.planner_step}): "
            f"action={step.action!r}, valid={step.action_valid}, "
            f"reward={step.reward:.3f}"
        )
        if feedback:
            line += f", feedback={feedback}"
        lines.append(line)

    return "\n".join(lines)


@dataclass
class SimpleUserPromptBuilder:
    """Domain-neutral user prompt builder.

    Use this for unit tests or for domains whose original prompt is already
    fully contained in the system message.  Real EM adapters should pass their
    own ``user_builder`` to reproduce the original prompt exactly.
    """

    max_history_steps: int = 20
    feedback_limit: int = 500
    first_step_suffix: str = "" # e.g. "This is the first planning step.Describe the current state and output an initial executable plan."
    replan_suffix: str = "" # e.g. "last plan failed / didn't complete the task, Use the interaction history and environment feedback"

    def __call__(
        self,
        spec: EpisodeSpec,
        observation: Observation,
        history: Sequence[TrajectoryStep],
        planner_step: int,
    ) -> str:
        parts = [f"## Instruction\n{spec.instruction}"]

        if observation.text.strip():
            parts.append(f"## Current Observation\n{observation.text.strip()}")

        history_text = format_history(
            history,
            max_steps=self.max_history_steps,
            feedback_limit=self.feedback_limit,
        )
        if history_text:
            parts.append(history_text)

        suffix = (
            self.first_step_suffix # first llm call
            if planner_step == 0
            else self.replan_suffix # replan (not first) llm call
        )
        if suffix.strip():
            parts.append(suffix.strip())

        return "\n\n".join(parts)

# [
#     {"role": "system", "content": system_prompt},
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": user_text},
#             {"type": "image_path", "path": "/path/frame.png"},
#         ],
#     },
# ]
# 图片先用 image_path 占位，等 backends/qwen_target.py 再转成：
# {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

@dataclass
class ContextBuilder:
    """Assemble the two messages sent to the target model.

    Parameters
    ----------
    base_system_prompt:
        The original EM system prompt, already formatted by the domain adapter
        with action space and few-shot examples.
    user_builder:
        Builds the user text.  Keeping this injectable lets each domain
        reproduce its original prompt without conditionals in the runner.
    image_key:
        Content-part type used for local image paths.  The target backend
        converts ``image_path`` parts to data URLs.
    """

    base_system_prompt: str
    user_builder: UserPromptBuilder
    image_key: str = "image_path"

    def build(
        self,
        spec: EpisodeSpec,
        observation: Observation,
        history: Sequence[TrajectoryStep],
        skill: str | None,
        *,
        planner_step: int,
    ) -> list[Message]:
        system = build_system_prompt(self.base_system_prompt, skill)
        user_text = self.user_builder(
            spec,
            observation,
            history,
            planner_step,
        )

        content: list[dict] = [
            {"type": "text", "text": user_text},
        ]
        for image_path in observation.images:
            content.append({
                "type": self.image_key,
                "path": image_path,
            })

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]


__all__ = [
    "ContextBuilder",
    "SimpleUserPromptBuilder",
    "UserPromptBuilder",
    "build_system_prompt",
    "format_history",
]