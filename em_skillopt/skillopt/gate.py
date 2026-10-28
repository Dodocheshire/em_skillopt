"""Validation gate for a candidate skill.

The aligned gate keeps both ``current`` and ``best`` state, supports hard /
soft / mixed scores, and can cache candidate evaluations by skill hash.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from ..core.metrics import summarize
from ..core.types import EpisodeResult, EpisodeSpec, EvalSummary
from ..harness.capability import RunnerFactory
from .rollout import RolloutConfig, rollout_specs

GateMetric = Literal["hard", "soft", "mixed"]


def _skill_hash(skill: str | None) -> str:
    text = (skill or "").strip()
    if not text:
        return ""
    return hashlib.sha256((text + "\n").encode("utf-8")).hexdigest()


def score_summary(
    summary: EvalSummary,
    *,
    metric: GateMetric = "hard",
    mixed_weight: float = 0.5,
) -> float:
    """Project a summary onto the gate comparison metric."""
    hard = float(summary.success_rate)
    soft = max(0.0, min(1.0, float(summary.mean_reward)))
    if metric == "hard":
        return hard
    if metric == "soft":
        return soft
    if metric == "mixed":
        weight = max(0.0, min(1.0, float(mixed_weight)))
        return (1.0 - weight) * hard + weight * soft
    raise ValueError(f"unknown gate metric: {metric!r}")


@dataclass
class GateResult:
    accepted: bool
    reason: str
    current: EvalSummary | None = None
    candidate: EvalSummary | None = None
    best: EvalSummary | None = None
    action: str = "reject"
    current_score: float = 0.0
    candidate_score: float = 0.0
    best_score: float = 0.0
    best_step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "action": self.action,
            "reason": self.reason,
            "current": self.current.to_dict() if self.current else None,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "best": self.best.to_dict() if self.best else None,
            "current_score": self.current_score,
            "candidate_score": self.candidate_score,
            "best_score": self.best_score,
            "best_step": self.best_step,
        }


def decide_gate(
    current: EvalSummary,
    candidate: EvalSummary,
    *,
    min_reward_delta: float = 0.0,
) -> tuple[bool, str]:
    """Backward-compatible hard success/reward decision."""

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
    workers: int = 1,
    proc_spec_factory=None,
) -> list[EpisodeResult]:
    return rollout_specs(
        specs,
        skill,
        out_dir=out_dir,
        runner_factory=runner_factory,
        proc_spec_factory=proc_spec_factory,
        config=RolloutConfig(
            resume=False,
            verbose=False,
            workers=workers,
        ),
    )


def gate_candidate(
    current_skill: str | None,
    candidate_skill: str,
    val_specs: Sequence[EpisodeSpec],
    *,
    runner_factory: RunnerFactory,
    out_dir: str | Path,
    min_reward_delta: float = 0.0,
    workers: int = 1,
    current_summary: EvalSummary | None = None,
    best_summary: EvalSummary | None = None,
    best_step: int = 0,
    metric: GateMetric = "hard",
    mixed_weight: float = 0.5,
    candidate_cache: dict[str, EvalSummary] | None = None,
    step: int = 0,
    proc_spec_factory=None,
) -> GateResult:
    """Evaluate current vs candidate, keeping best-so-far state."""

    if not val_specs:
        raise ValueError("gate requires at least one val episode")

    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if current_summary is None:
        print(f"[gate] current skill | val episodes={len(val_specs)}")
        current_results = _eval_skill(
            current_skill,
            val_specs,
            runner_factory=runner_factory,
            out_dir=root / "current",
            workers=workers,
            proc_spec_factory=proc_spec_factory,
        )
        current = summarize(
            current_results,
            env_name=val_specs[0].env_name,
            split="val",
        )
    else:
        print("[gate] reuse current val score")
        current = current_summary

    candidate_key = _skill_hash(candidate_skill)
    cached_candidate = (
        candidate_cache.get(candidate_key)
        if candidate_cache is not None and candidate_key
        else None
    )
    if cached_candidate is not None:
        print(f"[gate] candidate cache hit {candidate_key[:12]}")
        candidate = cached_candidate
    else:
        print(f"[gate] candidate skill | val episodes={len(val_specs)}")
        candidate_results = _eval_skill(
            candidate_skill,
            val_specs,
            runner_factory=runner_factory,
            out_dir=root / "candidate",
            workers=workers,
            proc_spec_factory=proc_spec_factory,
        )
        candidate = summarize(
            candidate_results,
            env_name=val_specs[0].env_name,
            split="val",
        )
        if candidate_cache is not None and candidate_key:
            candidate_cache[candidate_key] = candidate

    current_score = score_summary(
        current,
        metric=metric,
        mixed_weight=mixed_weight,
    )
    candidate_score = score_summary(
        candidate,
        metric=metric,
        mixed_weight=mixed_weight,
    )

    if best_summary is None:
        best = current
        best_score = current_score
        best_at = 0
    else:
        best = best_summary
        best_score = score_summary(
            best,
            metric=metric,
            mixed_weight=mixed_weight,
        )
        best_at = best_step

    if candidate.total <= 0:
        accepted = False
        action = "reject"
        reason = "candidate has no episodes"
    elif candidate_score > current_score:
        accepted = True
        if candidate_score > best_score:
            action = "accept_new_best"
            best = candidate
            best_score = candidate_score
            best_at = step
            reason = "new best"
        else:
            action = "accept"
            reason = "improved over current"
    else:
        accepted = False
        action = "reject"
        reason = "no improvement"

    result = GateResult(
        accepted=accepted,
        action=action,
        reason=reason,
        current=current,
        candidate=candidate,
        best=best,
        current_score=current_score,
        candidate_score=candidate_score,
        best_score=best_score,
        best_step=best_at,
    )
    (root / "gate.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


__all__ = [
    "GateMetric",
    "GateResult",
    "decide_gate",
    "gate_candidate",
    "score_summary",
]
