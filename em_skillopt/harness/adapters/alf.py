"""EB-ALFRED adapter.

Wraps ``embodiedbench.envs.eb_alfred.EBAlfEnv``.

Key differences from the other domains:
- ALFRED's action space is dynamic: ``env.language_skill_set`` changes after
  every ``reset()`` because object instances change per scene.
- ``env.step(action, reasoning)`` accepts an integer action id.
- ``info["last_action_success"]`` -> action_valid
- ``info["task_success"]`` -> success
"""
from __future__ import annotations

from typing import Any

from ...core.types import ActionValue, EpisodeSpec, Observation, StepResult
from .base import BaseAdapter


def _as_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


class AlfAdapter(BaseAdapter):
    def __init__(
        self,
        env: Any,
        *,
        log_path: str = "",
        max_steps: int | None = None,
    ) -> None:
        super().__init__(
            env,
            action_space=[],  # filled after the first reset()
            max_steps=max_steps,
            replan_on_invalid=True,
        )
        if log_path:
            self.env.log_path = log_path

    def reset(self, spec: EpisodeSpec) -> Observation:
        self.seek_episode(spec)
        self.env.reset()

        # EB-ALFRED rebuilds the action list for each scene.
        self.set_action_space(self.env.language_skill_set)

        images = _as_paths(self.env.save_image())
        return Observation(
            images=images,
            text=self.env.episode_language_instruction,
        )

    def step(self, action: ActionValue) -> StepResult:
        if not isinstance(action, int):
            raise TypeError(f"AlfAdapter expects int action, got {action!r}")

        obs, reward, done, info = self.env.step(action, reasoning="")

        images = _as_paths(self.env.save_image())
        feedback = str(info.get("env_feedback", ""))
        action_valid = bool(info.get("last_action_success", 1))
        success = bool(info.get("task_success", 0))

        return self.make_step_result(
            Observation(images=images, text=feedback),
            reward=float(reward),
            done=bool(done),
            success=success,
            feedback=feedback,
            action_valid=action_valid,
            info={
                "env_step": info.get("env_step"),
                "task_success": info.get("task_success"),
                "task_progress": info.get("task_progress"),
                "last_action_success": info.get("last_action_success"),
            },
        )