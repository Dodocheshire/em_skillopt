"""JitRL Method B plan adjustment (logit-free adaptation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..core.types import (
    ActionPlan,
    ActionValue,
    EpisodeResult,
    EpisodeSpec,
    Observation,
    TrajectoryStep,
)
from .memory import JitRLMemory, action_key, action_value


def compose_state(
    spec: EpisodeSpec,
    observation: Observation,
    history: Sequence[TrajectoryStep],
) -> str:
    """Text state used for memory retrieval."""

    parts = [spec.instruction, observation.text]
    for step in list(history)[-6:]:
        parts.append(
            f"{step.env_step}: action={step.action!r} "
            f"valid={step.action_valid} feedback={step.feedback}"
        )
    return "\n".join(part for part in parts if str(part).strip())


def _in_action_space(action: Any, action_space: Sequence[ActionValue]) -> bool:
    target = action_key(action)
    return any(action_key(candidate) == target for candidate in action_space)


@dataclass
class JitRLConfig:
    top_k: int = 5
    gamma: float = 0.95
    success_bonus: float = 1.0
    min_similarity: float = 0.0
    # If true, only reorder actions already in the target plan.
    reorder_only: bool = False
    # Only insert an action when its retrieval support reaches this count.
    min_support: int = 1
    # Episode weight, similar in spirit to JitRL's episode-dependent weight.
    episode_weight: float = 0.5


class JitRLPlanAdjuster:
    """Use retrieved action advantages to adjust each parsed plan.

    Official JitRL updates the token logits of candidate actions.  Since the
    remote target API does not expose logits, this adapter applies the same
    advantage estimate to the parsed action plan:

    1. retrieve similar (state, action, return) records;
    2. compute per-action advantage;
    3. reorder plan actions by advantage;
    4. optionally prepend a high-value memory action that is absent.

    This is intentionally a test-time method: no target weights are updated.
    """

    def __init__(self, memory: JitRLMemory, config: JitRLConfig | None = None) -> None:
        self.memory = memory
        self.config = config or JitRLConfig()

    def _query(
        self,
        spec: EpisodeSpec,
        observation: Observation,
        history: Sequence[TrajectoryStep],
    ) -> str:
        return compose_state(spec, observation, history)

    def adjust(
        self,
        plan: ActionPlan,
        spec: EpisodeSpec,
        observation: Observation,
        history: Sequence[TrajectoryStep],
        action_space: Sequence[ActionValue],
        planner_step: int,
    ) -> ActionPlan:
        if plan.error or not plan.actions or self.memory.size == 0:
            return plan

        query = self._query(spec, observation, history)
        advantages, support, baseline = self.memory.action_advantages(
            query,
            top_k=self.config.top_k,
            min_similarity=self.config.min_similarity,
        )
        if not advantages:
            return plan

        # Weight advantage similarly to JitRL's episode-dependent correction.
        weight = min(1.0 + self.config.episode_weight * planner_step, 1.5)
        scored = [
            (weight * advantages.get(action_key(action), 0.0), index, action)
            for index, action in enumerate(plan.actions)
        ]
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        adjusted = [action for _, _, action in scored]

        if not self.config.reorder_only:
            best_action = max(
                advantages.items(),
                key=lambda item: (item[1], support.get(item[0], 0)),
            )
            best_key, best_advantage = best_action
            if (
                best_advantage > 0.0
                and support.get(best_key, 0) >= self.config.min_support
                and all(action_key(action) != best_key for action in adjusted)
            ):
                candidate = action_value(best_key)
                if _in_action_space(candidate, action_space):
                    adjusted.insert(0, candidate)

        plan.actions = adjusted
        return plan

    def observe(self, result: EpisodeResult) -> None:
        """Store one completed episode in the cross-episode memory."""

        self.memory.add_episode(
            result,
            success_bonus=self.config.success_bonus,
        )


__all__ = [
    "JitRLConfig",
    "JitRLPlanAdjuster",
    "compose_state",
]
