"""Interfaces that keep the harness independent of EmbodiedBench and SkillOpt.

The runner only knows these protocols.  Concrete implementations live in:

- ``em_skillopt.harness.adapters``  -> EnvironmentAdapter
- ``em_skillopt.harness.context``   -> ContextBuilder
- ``em_skillopt.harness.parsing``   -> PlanParser
- ``em_skillopt.backends``          -> LLMClient
"""

from __future__ import annotations
from typing import Any, Protocol, Sequence, runtime_checkable

from .types import (
    ActionPlan,
    Message,
    LLMResponse,
    ActionValue,
    EpisodeSpec,
    Observation,
    StepResult,
    TrajectoryStep,
)

@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface for both the target and optimizer models.

    The target client receives multimodal messages.  The optimizer client
    receives text-only messages.  Both return ``LLMResponse``
    """
    def generate(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        ...

@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Thin wrapper around one EmbodiedBench environment.

    The adapter owns all simulator-specific details:

    - mapping ``EpisodeSpec.index`` to the environment's episode cursor
    - converting observations/images
    - calling the right ``step`` signature
    - deciding whether an invalid action should trigger replanning
    """
    @property
    def action_space(self) -> Sequence[ActionValue]:
        ...
    @property
    def max_steps(self) -> int:
        ...
    @property
    def replan_on_invalid(self) -> bool:
        ...
    
    def reset(self, spec: EpisodeSpec) -> Observation:
        ...

    def step(self, Action: ActionValue) -> StepResult:
        ...

    def save_image(self, observation: Observation, out_dir: str) -> list[str]:
        """Persist the current observation image(s), returning local paths."""

    def close(self) -> None:
        ...

@runtime_checkable
class ContextBuilder(Protocol):
    """Build the messages sent to the target model for one planning step."""

    def build(
        self,
        spec: EpisodeSpec,
        observation: Observation,
        history: Sequence[TrajectoryStep],
        skill: str | None,
        *,
        planner_step: int,
    ) -> list[Message]:
        ...


@runtime_checkable
class PlanParser(Protocol):
    """Parse one raw model response into an executable action plan."""

    def parse(self, raw: str, action_space: Sequence[ActionValue]) -> ActionPlan:
        ...

__all__ = [
    "ContextBuilder",
    "EnvironmentAdapter",
    "LLMClient",
    "PlanParser",
]
