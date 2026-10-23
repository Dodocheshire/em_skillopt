"""Aggregate EpisodeResult objects into evaluation metrics.

Two views are provided:

- ``summarize()``: typed ``EvalSummary`` used by training/evaluation code.
- ``em_summary()``: dict compatible with the original EM ``summary.json``.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from .types import EpisodeResult, EvalSummary


def summarize(
    results: Sequence[EpisodeResult],
    *,
    env_name: str,
    split: str,
) -> EvalSummary:
    """Typed summary used by the harness and SkillOpt bridge."""

    return EvalSummary.from_results(
        results,
        env_name=env_name,
        split=split,
    )


def em_summary(
    results: Sequence[EpisodeResult],
    *,
    env_name: str,
    split: str,
    save_path: str = "",
) -> dict[str, Any]:
    """Summary in the shape expected by existing EM report scripts."""

    total = len(results)
    successes = sum(1 for result in results if result.success)
    total_steps = sum(result.num_env_steps for result in results)
    total_planner_steps = sum(result.num_planner_steps for result in results)
    invalid_actions = sum(result.invalid_actions for result in results)
    format_error_episodes = sum(
        1 for result in results if result.format_errors > 0
    )
    elapsed = sum(result.elapsed_s for result in results)
    total_tokens = sum(
        result.token_usage.total_tokens for result in results
    )
    failure_reasons = Counter(
        result.failure_reason
        for result in results
        if result.failure_reason
    )

    return {
        "save_path": save_path,
        "env_name": env_name,
        "split": split,
        "total_num_tasks": total,
        "num_success": successes,
        "success_rate": successes / max(total, 1),
        "mean_reward": (
            sum(result.reward for result in results) / max(total, 1)
        ),
        "mean_num_steps": total_steps / max(total, 1),
        "avg_planner_steps": total_planner_steps / max(total, 1),
        "mean_invalid_actions": invalid_actions / max(total, 1),
        "mean_invalid_action_ratio": invalid_actions / max(total_steps, 1),
        "output_format_error": format_error_episodes,
        "mean_episode_elapsed_seconds": elapsed / max(total, 1),
        "total_tokens": total_tokens,
        "failure_reasons": dict(failure_reasons),
    }


def format_summary(summary: EvalSummary) -> str:
    """Human-readable one-block summary for terminal logs."""

    lines = [
        f"env={summary.env_name} split={summary.split}",
        f"success={summary.successes}/{summary.total} "
        f"({summary.success_rate:.4f})",
        f"mean_reward={summary.mean_reward:.4f}",
        f"mean_env_steps={summary.mean_env_steps:.2f}",
        f"mean_planner_steps={summary.mean_planner_steps:.2f}",
        f"invalid_action_rate={summary.invalid_action_rate:.4f}",
        f"format_error_rate={summary.format_error_rate:.4f}",
        f"elapsed_s={summary.elapsed_s:.1f}",
        f"tokens={summary.total_tokens}",
    ]

    if summary.per_capability:
        lines.append("per_capability:")
        for capability, bucket in sorted(summary.per_capability.items()):
            lines.append(
                f"  {capability}: "
                f"{bucket['successes']}/{bucket['total']} "
                f"({bucket['success_rate']:.4f})"
            )

    return "\n".join(lines)


__all__ = [
    "em_summary",
    "format_summary",
    "summarize",
]