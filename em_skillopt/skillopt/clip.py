"""Rank and clip merged edits.

This is the deterministic analogue of SkillOpt's ``optimizer/clip.py``: rank
edits by support and provenance, then keep at most ``max_edits`` edits.  The
edit budget is the "learning rate" of the skill update.
"""
from __future__ import annotations

import re
from typing import Any

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


def rank_and_clip(
    patch: dict[str, Any],
    *,
    max_edits: int = 3,
    current_skill: str | None = None,
) -> dict[str, Any]:
    """Sort edits and keep the top ``max_edits``.

    ``max_edits=0`` returns no edits.  A negative value keeps all edits.
    ``current_skill`` is accepted for API symmetry with SkillOpt; the current
    deterministic ranker does not read it.
    """

    del current_skill
    edits = [
        dict(edit)
        for edit in patch.get("edits", [])
        if isinstance(edit, dict)
    ]
    ranked = sorted(edits, key=_rank_key, reverse=True)
    selected = ranked if max_edits < 0 else ranked[:max_edits]

    result = dict(patch)
    result["edits"] = selected
    result["ranking_details"] = {
        "total_edits": len(ranked),
        "max_edits": max_edits,
        "selected_edits": len(selected),
    }
    return result


# Name used by the original SkillOpt API.
rank_and_select = rank_and_clip


__all__ = ["rank_and_clip", "rank_and_select"]
