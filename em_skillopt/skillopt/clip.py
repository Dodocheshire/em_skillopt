"""Rank and clip merged edits.

The aligned path asks the optimizer to rank edits against the current skill;
the deterministic support/source ranking remains as a fallback.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..harness.parsing import load_json_value

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SOURCE_PRIORITY = {"failure": 3, "mixed": 2, "success": 1}


def _tokens(text: Any) -> set[str]:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _rank_key(edit: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(edit.get("support_count", 1) or 1),
        _SOURCE_PRIORITY.get(str(edit.get("source_type", "failure")), 1),
        len(_tokens(edit.get("content", ""))),
        len(str(edit.get("target", ""))),
    )


def _llm_rank(
    llm: Any,
    current_skill: str | None,
    edits: list[dict[str, Any]],
    max_edits: int,
) -> list[dict[str, Any]] | None:
    lines = []
    for index, edit in enumerate(edits):
        lines.append(
            f"[{index}] op={edit.get('op')} "
            f"source={edit.get('source_type')} "
            f"support={edit.get('support_count', 1)} "
            f"target={edit.get('target', '')!r} "
            f"content={str(edit.get('content', ''))!r}"
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You rank skill edits by expected benefit for the current "
                "embodied agent. Return ONLY JSON: "
                '{"selected_indices": [0, 2, 1]}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Current skill:\n{current_skill or '(empty)'}\n\n"
                f"Candidate edits ({len(edits)} total, budget={max_edits}):\n"
                + "\n".join(lines)
                + f"\n\nSelect the {max_edits} most useful edits. "
                "Return their 0-based indices in priority order."
            ),
        },
    ]
    try:
        response = llm.generate(
            messages,
            max_tokens=4096,
            temperature=0.0,
            stage="ranking",
        )
        value = load_json_value(str(getattr(response, "text", response) or ""))
    except Exception:
        return None
    if not isinstance(value, dict):
        return None
    indices = value.get("selected_indices")
    if not isinstance(indices, list):
        return None
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index in indices:
        if (
            isinstance(index, int)
            and 0 <= index < len(edits)
            and index not in seen
        ):
            selected.append(edits[index])
            seen.add(index)
        if len(selected) >= max_edits:
            break
    return selected or None


def rank_and_clip(
    patch: dict[str, Any],
    *,
    max_edits: int = 3,
    current_skill: str | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Select edits, using LLM ranking when available.

    ``max_edits=0`` returns no edits.  A negative value keeps all edits.
    """

    edits = [
        dict(edit)
        for edit in patch.get("edits", [])
        if isinstance(edit, dict)
    ]
    ranked = sorted(edits, key=_rank_key, reverse=True)

    if max_edits < 0:
        selected = ranked
        method = "all"
    elif len(ranked) <= max_edits:
        selected = ranked
        method = "within_budget"
    elif max_edits == 0:
        selected = []
        method = "zero_budget"
    else:
        selected = _llm_rank(llm, current_skill, ranked, max_edits) if llm is not None else None
        if selected is None:
            selected = ranked[:max_edits]
            method = "deterministic_fallback"
        else:
            method = "llm"

    result = dict(patch)
    result["edits"] = selected
    result["ranking_details"] = {
        "total_edits": len(ranked),
        "max_edits": max_edits,
        "selected_edits": len(selected),
        "method": method,
    }
    return result


# Name used by the original SkillOpt API.
rank_and_select = rank_and_clip


__all__ = ["rank_and_clip", "rank_and_select"]
