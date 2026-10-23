"""EB-Navigation adapter.

Wraps ``embodiedbench.envs.eb_navigation.EBNavigationEnv``.

Key mappings:
- env.reset() -> Observation(images, instruction)
- env.step(action, reasoning, i_flag) -> StepResult
- info["last_action_success"] -> action_valid
- info["task_success"] -> success
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


class NavAdapter(BaseAdapter):
    def __init__(
        self,
        env: Any,
        *,
        log_path: str = "",
        max_steps: int | None = None,
    ) -> None:
        super().__init__(
            env,
            action_space=list(getattr(env, "language_skill_set", [])),
            max_steps=max_steps,
            replan_on_invalid=True,
        )
        if log_path:
            self.env.log_path = log_path

    def reset(self, spec: EpisodeSpec) -> Observation:
        self.seek_episode(spec)
        self.env.reset()
        images = _as_paths(self.env.save_image())
        return Observation(
            images=images,
            text=spec.instruction,
        )

    def step(self, action: ActionValue) -> StepResult:
        if not isinstance(action, int):
            raise TypeError(f"NavAdapter expects int action, got {action!r}")

        # i_flag only controls the env's per-step log formatting; it does not
        # affect reward/success. The runner steps one action at a time.
        obs, reward, done, info = self.env.step(
            action,
            reasoning="",
            i_flag=1,
        )

        images = _as_paths(self.env.save_image())
        feedback = str(info.get("env_feedback", ""))
        action_valid = bool(info.get("last_action_success", True))
        success = bool(info.get("task_success", reward > 0))

        return self.make_step_result(
            Observation(images=images, text=feedback),
            reward=float(reward),
            done=bool(done),
            success=success,
            feedback=feedback,
            action_valid=action_valid,
            info={
                "env_step": info.get("env_step"),
                "distance": info.get("distance"),
                "task_success": info.get("task_success"),
                "last_action_success": info.get("last_action_success"),
            },
        )