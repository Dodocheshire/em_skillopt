"""Core data types shared by the target harness and the SkillOpt bridge.

Keep this module small.  If a field is not read by the runner, an adapter, the
metrics aggregator, or the SkillOpt adapter, it does not belong here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence, TypeAlias

# int / str for high-level environments; list[int] for EB-Manipulation.
ActionValue: TypeAlias = int | str | list[int]

# OpenAI-style chat message.
Message: TypeAlias = dict[str, Any]


@dataclass
class TokenUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class EpisodeSpec:
    """One episode.

    ``index`` is the position in the original EmbodiedBench dataset.  The
    domain adapter uses it to move the simulator to the correct episode.
    """

    episode_id: str
    env_name: str
    index: int
    instruction: str
    capability: str = "base"
    split: str = "train"
    max_steps: int = 30
    seed: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """What the target model sees at one planning step."""

    images: list[str] = field(default_factory=list)
    text: str = ""


@dataclass
class ActionPlan:
    """Parsed model output for one planning step."""

    actions: list[ActionValue] = field(default_factory=list)
    raw: str = ""
    stop: bool = False
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None

    @property
    def is_empty(self) -> bool:
        return not self.actions


@dataclass
class StepResult:
    """Normalized result of one environment action."""

    observation: Observation
    reward: float = 0.0
    done: bool = False
    success: bool = False
    feedback: str = ""
    action_valid: bool = True
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanRecord:
    """One LLM call and its parsed plan, kept for reflection."""

    planner_step: int
    raw_response: str
    actions: list[ActionValue] = field(default_factory=list)
    error: str | None = None


@dataclass
class TrajectoryStep:
    """One executed action, kept for reflection and debugging."""

    planner_step: int
    env_step: int
    action: ActionValue
    action_valid: bool
    reward: float
    feedback: str
    done: bool
    success: bool


@dataclass
class EpisodeResult:
    """Result of one episode."""

    spec: EpisodeSpec
    success: bool = False
    reward: float = 0.0
    num_env_steps: int = 0
    num_planner_steps: int = 0
    invalid_actions: int = 0
    format_errors: int = 0
    elapsed_s: float = 0.0
    failure_reason: str = ""
    plans: list[PlanRecord] = field(default_factory=list)
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def hard(self) -> float:
        return 1.0 if self.success else 0.0

    @property
    def soft_score(self) -> float:
        # ALF/Habitat/Nav rewards are 0/1; keep this clipped for dense rewards.
        return max(0.0, min(1.0, float(self.reward)))

    @property
    def empty_plans(self) -> int:
        return sum(1 for plan in self.plans if not plan.actions)

    def to_rollout_result(self) -> dict[str, Any]:
        """Dict shape expected by SkillOpt's EnvAdapter.rollout()."""

        return {
            "id": self.spec.episode_id,
            "hard": self.hard,
            "soft": self.soft_score,
            "n_turns": self.num_env_steps,
            "fail_reason": self.failure_reason,
            "task_type": self.spec.capability,
            "env": self.spec.env_name,
            "instruction": self.spec.instruction,
            "num_planner_steps": self.num_planner_steps,
            "invalid_actions": self.invalid_actions,
            "format_errors": self.format_errors,
            "elapsed_s": self.elapsed_s,
            "token_usage": asdict(self.token_usage),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalSummary:
    """Aggregated metrics for one environment/split/method."""

    env_name: str
    split: str
    total: int = 0
    successes: int = 0
    success_rate: float = 0.0
    mean_reward: float = 0.0
    mean_env_steps: float = 0.0
    mean_planner_steps: float = 0.0
    invalid_action_rate: float = 0.0
    format_error_rate: float = 0.0
    elapsed_s: float = 0.0
    total_tokens: int = 0
    per_capability: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_results(
        cls,
        results: Sequence[EpisodeResult],
        *,
        env_name: str,
        split: str,
    ) -> "EvalSummary":
        total = len(results)
        successes = sum(result.success for result in results)
        total_env_steps = sum(result.num_env_steps for result in results)
        total_planner_steps = sum(result.num_planner_steps for result in results)
        invalid_actions = sum(result.invalid_actions for result in results)
        format_errors = sum(result.format_errors for result in results)
        elapsed_s = sum(result.elapsed_s for result in results)
        total_tokens = sum(result.token_usage.total_tokens for result in results)

        per_capability: dict[str, dict[str, Any]] = {}
        for result in results:
            capability = result.spec.capability
            bucket = per_capability.setdefault(
                capability,
                {
                    "total": 0,
                    "successes": 0,
                    "success_rate": 0.0,
                    "mean_reward": 0.0,
                },
            )
            bucket["total"] += 1
            bucket["successes"] += int(result.success)
            bucket["mean_reward"] += result.reward

        for bucket in per_capability.values():
            count = max(bucket["total"], 1)
            bucket["success_rate"] = bucket["successes"] / count
            bucket["mean_reward"] = bucket["mean_reward"] / count

        return cls(
            env_name=env_name,
            split=split,
            total=total,
            successes=successes,
            success_rate=successes / max(total, 1),
            mean_reward=sum(result.reward for result in results) / max(total, 1),
            mean_env_steps=total_env_steps / max(total, 1),
            mean_planner_steps=total_planner_steps / max(total, 1),
            invalid_action_rate=invalid_actions / max(total_env_steps, 1),
            format_error_rate=format_errors / max(total, 1),
            elapsed_s=elapsed_s,
            total_tokens=total_tokens,
            per_capability=per_capability,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "ActionPlan",
    "ActionValue",
    "EpisodeResult",
    "EpisodeSpec",
    "EvalSummary",
    "LLMResponse",
    "Message",
    "Observation",
    "PlanRecord",
    "StepResult",
    "TokenUsage",
    "TrajectoryStep",
]