"""Validation gate for a candidate skill.

The gate runs current and candidate skills on the same held-out val specs and
accepts the candidate only when it improves success rate (or keeps it and
improves reward).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..core.metrics import summarize
from ..core.types import EpisodeResult, EpisodeSpec, EvalSummary
from ..harness.capability import RunnerFactory
from .rollout import RolloutConfig, rollout_specs


@dataclass
class GateResult:
    accepted: bool
    reason: str
    current: EvalSummary | None = None
    candidate: EvalSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "current": self.current.to_dict() if self.current else None,
            "candidate": self.candidate.to_dict() if self.candidate else None,
        }


def decide_gate(
    current: EvalSummary,
    candidate: EvalSummary,
    *,
    min_reward_delta: float = 0.0,
) -> tuple[bool, str]:
    """Pure accept/reject decision, useful for tests."""

    if candidate.total <= 0:
        return False, "candidate has no episodes"
    if candidate.success_rate > current.success_rate:
        return True, "success_rate improved"
    if (
        candidate.success_rate == current.success_rate
        and candidate.mean_reward > current.mean_reward + min_reward_delta
    ):
        return True, "reward improved at equal success_rate"
    return False, "no improvement"


def _eval_skill(
    skill: str | None,
    specs: Sequence[EpisodeSpec],
    *,
    runner_factory: RunnerFactory,
    out_dir: Path,
) -> list[EpisodeResult]:
    return rollout_specs(
        specs,
        skill,
        out_dir=out_dir,
        runner_factory=runner_factory,
        config=RolloutConfig(resume=False, verbose=False),
    )


def gate_candidate(
    current_skill: str | None,
    candidate_skill: str,
    val_specs: Sequence[EpisodeSpec],
    *,
    runner_factory: RunnerFactory,
    out_dir: str | Path,
    min_reward_delta: float = 0.0,
) -> GateResult:
    """Evaluate current vs candidate on the same val specs."""

    if not val_specs:
        raise ValueError("gate requires at least one val episode")

    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    print(f"[gate] current skill | val episodes={len(val_specs)}")
    current_results = _eval_skill(
        current_skill,
        val_specs,
        runner_factory=runner_factory,
        out_dir=root / "current",
    )
    print(f"[gate] candidate skill | val episodes={len(val_specs)}")
    candidate_results = _eval_skill(
        candidate_skill,
        val_specs,
        runner_factory=runner_factory,
        out_dir=root / "candidate",
    )

    env_name = val_specs[0].env_name
    current = summarize(current_results, env_name=env_name, split="val")
    candidate = summarize(candidate_results, env_name=env_name, split="val")
    accepted, reason = decide_gate(
        current,
        candidate,
        min_reward_delta=min_reward_delta,
    )
    result = GateResult(
        accepted=accepted,
        reason=reason,
        current=current,
        candidate=candidate,
    )
    (root / "gate.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


__all__ = ["GateResult", "decide_gate", "gate_candidate"]
