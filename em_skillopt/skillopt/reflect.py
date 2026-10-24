"""Turn EM episodes into skill patches.

This mirrors SkillOpt's minibatch Reflect stage more closely than the first
prototype:

* failures and successes are analysed in separate optimizer calls;
* the full plan text and the full action trajectory are included, not a
  three-plan / tail-only digest;
* previous-step failure patterns and rejected edits can be supplied as context;
* the optimizer gets a real edit budget (L) instead of a hard-coded 3.
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
Return at most {max_edits} edits."""


@dataclass
class ReflectConfig:
    max_episodes: int = 8
    max_failures: int = 8
    max_successes: int = 8
    # 0 means no truncation.  SkillOpt gives the analyst the full trajectory.
    max_plan_chars: int = 0
    max_tokens: int = 16384
    max_edits: int = 3


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if limit <= 0 else text[:limit]


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
        "feedback": _truncate(step.feedback, 0),
        "done": step.done,
    }


def episode_digest(
    result: EpisodeResult,
    *,
    max_plan_chars: int = 0,
) -> dict[str, Any]:
    """Compress one EpisodeResult into the text reflection input.

    The full plan and trajectory are retained by default.  ``max_plan_chars``
    is available for cost control but is 0 (unlimited) in the aligned path.
    """

    plans = result.plans
    trajectory = [
        _trajectory_digest(step) for step in result.trajectory
    ]
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
        "target_system_prompt": _truncate(result.target_system_prompt, 0),
        "target_user_prompt": _truncate(result.target_user_prompt, 0),
        "plans": [
            _plan_digest(plan, max_plan_chars) for plan in plans
        ],
        "trajectory": trajectory,
        # Backward-compatible key used by the old prototype/tests.
        "trajectory_tail": trajectory,
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
    *,
    group: str,
    context: str,
    max_edits: int,
) -> list[dict[str, Any]]:
    payload = json.dumps(list(digests), ensure_ascii=False, indent=2)
    user = (
        "Current skill:\n"
        f"{current_skill or '(empty)'}\n\n"
        f"Episode group: {group}\n"
        f"{'Previous-step context:' + chr(10) + context + chr(10) if context.strip() else ''}"
        "Episodes:\n"
        f"{payload}\n\n"
        f"Produce at most {max_edits} edits for this group.\n"
        "Return the JSON patch object now."
    )
    return [
        {
            "role": "system",
            "content": _SYSTEM.replace("{max_edits}", str(max_edits)),
        },
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


def _normalize_patch(raw: str, max_edits: int) -> dict[str, Any]:
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
    for item in value.get("edits", [])[:max_edits]:
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


def _reflect_group(
    group: str,
    results: Sequence[EpisodeResult],
    current_skill: str | None,
    *,
    llm: Any,
    out_dir: Path,
    cfg: ReflectConfig,
    context: str,
) -> dict[str, Any] | None:
    if not results:
        return None

    digests = [
        episode_digest(result, max_plan_chars=cfg.max_plan_chars)
        for result in results
    ]
    (out_dir / f"digests_{group}.json").write_text(
        json.dumps({"group": group, "digests": digests}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    text = _call_llm(
        llm,
        _build_messages(
            current_skill,
            digests,
            group=group,
            context=context,
            max_edits=cfg.max_edits,
        ),
        cfg,
    )
    patch = _normalize_patch(text, cfg.max_edits)
    patch["source_type"] = group
    patch["batch_size"] = len(results)
    patch["group"] = group
    return patch


def reflect_episodes(
    results: Sequence[EpisodeResult],
    current_skill: str | None,
    *,
    llm: Any,
    out_dir: str | Path,
    config: ReflectConfig | None = None,
    context: str = "",
) -> list[dict[str, Any]]:
    """Run separate failure/success reflection calls for one minibatch."""

    if not results:
        raise ValueError("reflect requires at least one episode result")

    cfg = config or ReflectConfig()
    selected = _select(results, cfg)
    failures = [result for result in selected if not result.success]
    successes = [result for result in selected if result.success]

    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    patches: list[dict[str, Any]] = []
    for group, group_results in (("failure", failures), ("success", successes)):
        patch = _reflect_group(
            group,
            group_results,
            current_skill,
            llm=llm,
            out_dir=root,
            cfg=cfg,
            context=context,
        )
        if patch is not None:
            patches.append(patch)

    if not patches:
        raise ValueError("reflection produced no groups")

    (root / "digests.json").write_text(
        json.dumps(
            {
                "digests": [
                    episode_digest(
                        result,
                        max_plan_chars=cfg.max_plan_chars,
                    )
                    for result in selected
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
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
