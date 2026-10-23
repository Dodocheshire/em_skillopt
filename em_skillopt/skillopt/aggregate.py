"""Merge reflection patches into one editable patch.

This is a deterministic first version of SkillOpt's aggregate stage.  It
deduplicates edits, merges lexically similar edits, tracks support counts, and
resolves simple conflicts with failure priority.  No extra LLM call is used.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NEGATIONS = {"not", "never", "avoid", "without", "no"}
_ALLOWED_OPS = {"append", "insert_after", "replace", "delete"}
_SOURCE_PRIORITY = {"failure": 3, "mixed": 2, "success": 1}
_SIMILARITY_THRESHOLD = 0.75


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

    # Two replacements/deletes on the same target with different content.
    if left_target and left_target == right_target:
        if left.get("op") != right.get("op"):
            return True
        if left.get("op") in {"replace", "delete"} and left_content != right_content:
            return True

    # "rotate" vs "do not rotate" / "avoid rotate".
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
        "support_count": max(1, int(edit.get("support_count") or 1)),
        "merged_from": 1,
    }


def merge_patches(
    patches: Sequence[dict[str, Any]],
    *,
    similarity_threshold: float = _SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """Deduplicate, merge similar edits, and resolve simple conflicts."""

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
            conflicts.append({
                "kept": conflict_with,
                "dropped": item,
            })
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
        "reasoning": (
            f"merged {len(candidates)} edits into {len(merged)} edits"
        ),
        "edits": merged,
        "conflicts": conflicts,
        "num_input_edits": len(candidates),
        "num_merged_edits": len(merged),
    }


__all__ = ["merge_patches"]
