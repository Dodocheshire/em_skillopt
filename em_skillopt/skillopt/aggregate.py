"""Aggregate reflection patches.

The aligned path mirrors SkillOpt: failure patches and success patches are
merged separately by the optimizer, hierarchically, with failures taking
priority.  A deterministic lexical merge remains as a fallback when no LLM is
supplied (unit tests and degraded runs).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from ..harness.parsing import load_json_value

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NEGATIONS = {"not", "never", "avoid", "without", "no"}
_ALLOWED_OPS = {"append", "insert_after", "replace", "delete"}
_SOURCE_PRIORITY = {"failure": 3, "mixed": 2, "success": 1}
_SIMILARITY_THRESHOLD = 0.75

_MERGE_SYSTEM = """You merge skill-update patches for an embodied agent.
Return ONLY a JSON object:
{"reasoning": "...", "edits": [
  {"op": "append", "content": "...", "target": "", "source_type": "failure"}
]}
Keep concrete, reusable rules. Remove duplicates and contradictory rules.
Prefer failure-driven edits when two edits conflict. Preserve support counts
when possible. Do not mention episode IDs."""


def _norm(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _tokens(text: Any) -> set[str]:
    return set(_TOKEN_RE.findall(_norm(text)))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _source_priority(source: Any) -> int:
    return _SOURCE_PRIORITY.get(str(source or "failure"), 1)


def _score(edit: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(edit.get("support_count", 1) or 1),
        _source_priority(edit.get("source_type")),
        len(_tokens(edit.get("content", ""))),
    )


def _is_similar(
    left: dict[str, Any],
    right: dict[str, Any],
    threshold: float = _SIMILARITY_THRESHOLD,
) -> bool:
    if left.get("op") != right.get("op"):
        return False
    if _norm(left.get("content")) == _norm(right.get("content")):
        return True
    return _jaccard(
        _tokens(left.get("content", "")),
        _tokens(right.get("content", "")),
    ) >= threshold


def _is_conflict(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_content = _norm(left.get("content"))
    right_content = _norm(right.get("content"))
    left_target = _norm(left.get("target"))
    right_target = _norm(right.get("target"))

    if left_target and left_target == right_target:
        if left.get("op") != right.get("op"):
            return True
        if left.get("op") in {"replace", "delete"} and left_content != right_content:
            return True

    left_tokens = _tokens(left_content)
    right_tokens = _tokens(right_content)
    left_neg = bool(left_tokens & _NEGATIONS)
    right_neg = bool(right_tokens & _NEGATIONS)
    if left_neg != right_neg:
        left_pos = left_tokens - _NEGATIONS
        right_pos = right_tokens - _NEGATIONS
        if _jaccard(left_pos, right_pos) >= 0.6:
            return True
    return False


def _candidate(edit: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any] | None:
    op = str(edit.get("op", "append"))
    content = str(edit.get("content", "")).strip()
    target = str(edit.get("target", "")).strip()
    if op not in _ALLOWED_OPS:
        return None
    if op != "delete" and not content:
        return None
    if op != "append" and not target:
        return None
    return {
        "op": op,
        "content": content,
        "target": target,
        "source_type": str(edit.get("source_type") or patch.get("source_type") or "failure"),
        "support_count": max(1, int(edit.get("support_count") or patch.get("batch_size") or 1)),
        "merged_from": 1,
    }


def _deterministic_merge(
    patches: Sequence[dict[str, Any]],
    *,
    similarity_threshold: float = _SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for patch in patches:
        inner = patch.get("patch", patch)
        if not isinstance(inner, dict):
            continue
        for edit in inner.get("edits", []):
            if isinstance(edit, dict):
                item = _candidate(edit, patch)
                if item is not None:
                    candidates.append(item)

    candidates.sort(key=_score, reverse=True)
    merged: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for item in candidates:
        conflict_with = next(
            (existing for existing in merged if _is_conflict(item, existing)),
            None,
        )
        if conflict_with is not None:
            conflicts.append({"kept": conflict_with, "dropped": item})
            continue

        similar_to = next(
            (
                existing for existing in merged
                if _is_similar(item, existing, similarity_threshold)
            ),
            None,
        )
        if similar_to is not None:
            similar_to["support_count"] += item["support_count"]
            similar_to["merged_from"] += 1
            continue

        merged.append(item)

    merged.sort(key=_score, reverse=True)
    return {
        "reasoning": f"merged {len(candidates)} edits into {len(merged)} edits",
        "edits": merged,
        "conflicts": conflicts,
        "num_input_edits": len(candidates),
        "num_merged_edits": len(merged),
    }


def _patch_edits(patch: dict[str, Any]) -> list[dict[str, Any]]:
    inner = patch.get("patch", patch)
    if not isinstance(inner, dict):
        return []
    return [edit for edit in inner.get("edits", []) if isinstance(edit, dict)]


def _llm_merge(
    llm: Any,
    current_skill: str | None,
    patches: Sequence[dict[str, Any]],
    *,
    group: str,
    max_tokens: int = 8192,
) -> dict[str, Any] | None:
    payload = json.dumps(list(patches), ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": _MERGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Current skill:\n{current_skill or '(empty)'}\n\n"
                f"Group: {group}\n"
                f"Patches to merge ({len(patches)}):\n{payload}\n\n"
                "Return the merged JSON patch now."
            ),
        },
    ]
    try:
        response = llm.generate(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            stage="merge",
        )
        value = load_json_value(str(getattr(response, "text", response) or ""))
    except Exception:
        return None
    if not isinstance(value, dict):
        return None
    edits = value.get("edits")
    if not isinstance(edits, list):
        return None
    normalized: list[dict[str, Any]] = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        item = _candidate(edit, {"source_type": group})
        if item is not None:
            normalized.append(item)
    if not normalized:
        return None
    return {
        "reasoning": str(value.get("reasoning", "")),
        "edits": normalized,
        "source_type": group,
        "num_merged_edits": len(normalized),
    }


def _hierarchical_merge(
    llm: Any,
    current_skill: str | None,
    patches: Sequence[dict[str, Any]],
    *,
    group: str,
    merge_batch_size: int,
) -> dict[str, Any]:
    if not patches:
        return {"reasoning": "no patches", "edits": [], "source_type": group}
    current = list(patches)
    level = 0
    while len(current) > 1:
        level += 1
        next_level: list[dict[str, Any]] = []
        for index in range(0, len(current), max(2, merge_batch_size)):
            batch = current[index:index + max(2, merge_batch_size)]
            if len(batch) == 1:
                next_level.append(batch[0])
                continue
            merged = _llm_merge(
                llm,
                current_skill,
                batch,
                group=group,
            )
            if merged is None:
                # Fallback for one failed level: concatenate the batch.
                merged = _deterministic_merge(batch)
                merged["source_type"] = group
                merged["fallback"] = True
            merged["merge_level"] = level
            next_level.append(merged)
        current = next_level
    return current[0]


def merge_patches(
    patches: Sequence[dict[str, Any]],
    *,
    similarity_threshold: float = _SIMILARITY_THRESHOLD,
    llm: Any | None = None,
    current_skill: str | None = None,
    merge_batch_size: int = 8,
) -> dict[str, Any]:
    """LLM failure-first hierarchical merge, with deterministic fallback."""

    if llm is None:
        return _deterministic_merge(
            patches,
            similarity_threshold=similarity_threshold,
        )

    failures = [
        patch for patch in patches
        if str(patch.get("source_type", "failure")) != "success"
    ]
    successes = [
        patch for patch in patches
        if str(patch.get("source_type", "failure")) == "success"
    ]

    failure_merged = _hierarchical_merge(
        llm,
        current_skill,
        failures,
        group="failure",
        merge_batch_size=merge_batch_size,
    )
    success_merged = _hierarchical_merge(
        llm,
        current_skill,
        successes,
        group="success",
        merge_batch_size=merge_batch_size,
    )

    f_edits = _patch_edits(failure_merged)
    s_edits = _patch_edits(success_merged)
    if f_edits and s_edits:
        final = _llm_merge(
            llm,
            current_skill,
            [failure_merged, success_merged],
            group="final",
        )
        if final is None:
            final = _deterministic_merge([failure_merged, success_merged])
            final["fallback"] = True
    elif f_edits:
        final = failure_merged
    elif s_edits:
        final = success_merged
    else:
        final = {"reasoning": "no edits", "edits": []}

    edits = _patch_edits(final)
    result = dict(final)
    result["edits"] = edits
    result["num_input_edits"] = sum(len(_patch_edits(patch)) for patch in patches)
    result["num_merged_edits"] = len(edits)
    return result


__all__ = ["merge_patches"]
