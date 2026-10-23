"""EB-Habitat adapter.

Wraps ``embodiedbench.envs.eb_habitat.EBHabEnv``.

Habitat is special: its ``reset()`` advances an internal episode iterator.
It does not support jumping backwards by simply setting
``_current_episode_num``.  We therefore seek forward by calling ``reset()``
until the requested index is reached; backwards seeks raise an error.

If a training split needs random episode order, construct a new EBHabEnv with
``start_epi_index`` or run Habitat episodes in ascending index order.
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


class HabAdapter(BaseAdapter):
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
        self._last_obs: Any = None

    def reset(self, spec: EpisodeSpec) -> Observation:
        """Load ``spec.index`` by advancing Habitat's internal iterator."""

        while self.env._current_episode_num < spec.index:
            self.env.reset()
            self._last_obs = None

        if self.env._current_episode_num > spec.index:
            raise RuntimeError(
                "Habitat cannot seek backwards: "
                f"current={self.env._current_episode_num}, "
                f"requested={spec.index}. Use ascending episode order "
                "or construct a new EBHabEnv with start_epi_index."
            )

        obs = self.env.reset()
        self._last_obs = obs

        images = _as_paths(self.env.save_image(obs))
        return Observation(
            images=images,
            text=self.env.episode_language_instruction,
        )

    def step(self, action: ActionValue) -> StepResult:
        if not isinstance(action, int):
            raise TypeError(f"HabAdapter expects int action, got {action!r}")

        obs, reward, done, info = self.env.step(action, reasoning="")
        self._last_obs = obs

        images = _as_paths(self.env.save_image(obs))
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