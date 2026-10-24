"""Turn EM episodes into small skill patches.

This is the text-only reflection stage of Method A.  Images are not sent to the
optimizer; each episode is compressed to a short digest with its key planning
rounds and environment feedback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..core.types import EpisodeResult, PlanRecord, TrajectoryStep
from ..harness.parsing import load_json_value

_SYSTEM = """You improve a reusable skill for an embodied agent.
Analyze the episodes and return ONLY a JSON object:
{"reasoning": "...", "edits": [
  {"op": "append", "content": "...", "source_type": "failure"}
]}
Use op append, insert_after, replace, or delete.
Edits must be short, actionable, reusable rules. Do not mention episode IDs.
Return at most 3 edits."""


@dataclass
class ReflectConfig:
    max_episodes: int = 8
    max_failures: int = 6
    max_successes: int = 2
    max_plan_chars: int = 2000
    max_tokens: int = 2048


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "").strip().replace("\n", " ")
    return text[:limit]


def _plan_digest(plan: PlanRecord, limit: int) -> dict[str, Any]:
    return {
        "planner_step": plan.planner_step,
        "actions": plan.actions,
        "error": plan.error,
        "response": _truncate(plan.raw_response, limit),
    }


def _trajectory_digest(step: TrajectoryStep) -> dict[str, Any]:
    return {
        "env_step": step.env_step,
        "action": step.action,
        "valid": step.action_valid,
        "reward": step.reward,
        "feedback": _truncate(step.feedback, 300),
        "done": step.done,
    }


def episode_digest(
    result: EpisodeResult,
    *,
    max_plan_chars: int = 2000,
) -> dict[str, Any]:
    """Compress one EpisodeResult into the text reflection input."""

    plans = result.plans
    # First, last, and the first error/invalid round are the most useful.
    selected: list[int] = []
    if plans:
        selected.append(0)
        if len(plans) > 1:
            selected.append(len(plans) - 1)
        error_idx = next(
            (i for i, plan in enumerate(plans) if plan.error),
            None,
        )
        if error_idx is not None and error_idx not in selected:
            selected.append(error_idx)

    return {
        "id": result.spec.episode_id,
        "env": result.spec.env_name,
        "capability": result.spec.capability,
        "instruction": result.spec.instruction,
        "success": result.success,
        "reward": result.reward,
        "env_steps": result.num_env_steps,
        "planner_steps": result.num_planner_steps,
        "invalid_actions": result.invalid_actions,
        "format_errors": result.format_errors,
        "failure_reason": result.failure_reason,
        "plans": [
            _plan_digest(plans[i], max_plan_chars) for i in selected
        ],
        "trajectory_tail": [
            _trajectory_digest(step) for step in result.trajectory[-5:]
        ],
    }


def _select(results: Sequence[EpisodeResult], cfg: ReflectConfig) -> list[EpisodeResult]:
    failures = [r for r in results if not r.success]
    successes = [r for r in results if r.success]
    return (
        failures[: cfg.max_failures]
        + successes[: cfg.max_successes]
    )[: cfg.max_episodes]


def _build_messages(
    current_skill: str | None,
    digests: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    payload = json.dumps(list(digests), ensure_ascii=False, indent=2)
    user = (
        "Current skill:\n"
        f"{current_skill or '(empty)'}\n\n"
        "Episodes:\n"
        f"{payload}\n\n"
        "Return the JSON patch object now."
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def _call_llm(llm: Any, messages: list[dict[str, Any]], cfg: ReflectConfig) -> str:
    response = llm.generate(
        messages,
        max_tokens=cfg.max_tokens,
        temperature=0.0,
        stage="reflect",
    )
    return str(getattr(response, "text", response) or "")


def _normalize_patch(raw: str) -> dict[str, Any]:
    try:
        value = load_json_value(raw)
    except Exception as exc:  # malformed optimizer output
        return {
            "reasoning": "",
            "edits": [],
            "parse_error": str(exc),
        }

    if not isinstance(value, dict):
        return {"reasoning": "", "edits": [], "parse_error": "not an object"}

    edits: list[dict[str, Any]] = []
    # The reflection prompt allows at most 3 edits per minibatch.
    for item in value.get("edits", [])[:3]:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op", "append"))
        content = str(item.get("content", "")).strip()
        if op not in {"append", "insert_after", "replace", "delete"}:
            continue
        if op != "delete" and not content:
            continue
        edits.append({
            "op": op,
            "content": content,
            "target": str(item.get("target", "")),
            "source_type": str(item.get("source_type", "failure")),
        })

    return {
        "reasoning": str(value.get("reasoning", "")),
        "edits": edits,
    }


def reflect_episodes(
    results: Sequence[EpisodeResult],
    current_skill: str | None,
    *,
    llm: Any,
    out_dir: str | Path,
    config: ReflectConfig | None = None,
) -> list[dict[str, Any]]:
    """Run one reflection minibatch and return one patch.

    Current: 4-8 episodes -> one optimizer call -> one patch. Extra results
    beyond the configured limits are ignored; aggregation, skill update and
    validation are left to the caller.

    Future: a large rollout batch is split into multiple minibatches, this
    function is called per minibatch, then patches are aggregated, ranked,
    applied and validated.
    """

    if not results:
        raise ValueError("reflect requires at least one episode result")

    cfg = config or ReflectConfig()
    selected = _select(results, cfg)
    digests = [episode_digest(r, max_plan_chars=cfg.max_plan_chars) for r in selected]

    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "digests.json").write_text(
        json.dumps({"digests": digests}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    text = _call_llm(llm, _build_messages(current_skill, digests), cfg)
    patch = _normalize_patch(text)
    failures = sum(1 for result in selected if not result.success)
    successes = sum(1 for result in selected if result.success)
    if failures and successes:
        patch["source_type"] = "mixed"
    elif successes:
        patch["source_type"] = "success"
    else:
        patch["source_type"] = "failure"
    patch["batch_size"] = len(selected)
    patches = [patch]

    (root / "patches.json").write_text(
        json.dumps({"patches": patches}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return patches


__all__ = [
    "ReflectConfig",
    "episode_digest",
    "reflect_episodes",
]
