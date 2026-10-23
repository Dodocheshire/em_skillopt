"""Shared helpers for the four EmbodiedBench domain adapters.

The runner calls only the ``EnvironmentAdapter`` protocol.  This base class
centralizes the pieces that are identical across domains:

- episode index seeking
- action-space storage
- max-step lookup
- close
- StepResult construction
"""
from __future__ import annotations

from typing import Any, Sequence

from ...core.types import (
    ActionValue,
    EpisodeSpec,
    Observation,
    StepResult,
)


class BaseAdapter:
    """Minimal shared implementation for domain adapters."""

    def __init__(
        self,
        env: Any,
        *,
        action_space: Sequence[ActionValue] | None = None,
        max_steps: int | None = None,
        replan_on_invalid: bool = True,
    ) -> None:
        self.env = env
        self.replan_on_invalid = replan_on_invalid
        self._max_steps = (
            max_steps
            or getattr(env, "_max_episode_steps", None)
            or 30
        )
        self._action_space: list[ActionValue] = list(action_space or [])

    @property
    def action_space(self) -> Sequence[ActionValue]:
        return self._action_space

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def set_action_space(self, action_space: Sequence[ActionValue]) -> None:
        """Used by dynamic-action-space environments such as EB-ALFRED."""

        self._action_space = list(action_space)

    def seek_episode(self, spec: EpisodeSpec) -> None:
        """Move ``env._current_episode_num`` to the requested episode.

        Navigation, ALFRED and Manipulation all increment the cursor inside
        ``reset()``, so the caller must set it to ``spec.index`` first.
        Habitat overrides this because its internal iterator advances itself.
        """

        if not hasattr(self.env, "_current_episode_num"):
            raise AttributeError(
                f"{type(self.env).__name__} has no _current_episode_num; "
                "override seek_episode() in the domain adapter"
            )
        self.env._current_episode_num = spec.index

    def reset(self, spec: EpisodeSpec) -> Observation:
        raise NotImplementedError

    def step(self, action: ActionValue) -> StepResult:
        raise NotImplementedError

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    @staticmethod
    def make_step_result(
        observation: Observation,
        *,
        reward: float = 0.0,
        done: bool = False,
        success: bool = False,
        feedback: str = "",
        action_valid: bool = True,
        info: dict[str, Any] | None = None,
    ) -> StepResult:
        return StepResult(
            observation=observation,
            reward=float(reward),
            done=bool(done), # episode 是否终止
            success=bool(success), # episode 任务是否成功
            feedback=feedback,
            action_valid=bool(action_valid),
            info=dict(info or {}),
        )


__all__ = ["BaseAdapter"]
    

