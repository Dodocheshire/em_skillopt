"""Apply reflection patches to the current skill.

Update is deliberately simple: patches contain a few text edits, and the
validation gate decides whether the resulting candidate is kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

DEFAULT_MAX_CHARS = 6000
_ALLOWED_OPS = {"append", "insert_after", "replace", "delete"}


@dataclass
class UpdateResult:
    content: str
    applied: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)


def _edits_from_patch(patch: dict[str, Any]) -> list[dict[str, Any]]:
    inner = patch.get("patch", patch)
    if not isinstance(inner, dict):
        return []
    edits = inner.get("edits", [])
    return [edit for edit in edits if isinstance(edit, dict)]


def _contains(text: str, content: str) -> bool:
    return bool(content) and content in text


def apply_patches(
    current_skill: str | None,
    patches: Sequence[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> UpdateResult:
    """Return the candidate skill after applying valid edits."""

    text = (current_skill or "").strip()
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for patch in patches:
        for edit in _edits_from_patch(patch):
            op = str(edit.get("op", "append"))
            content = str(edit.get("content", "")).strip()
            target = str(edit.get("target", "")).strip()

            if op not in _ALLOWED_OPS:
                skipped.append({"op": op, "reason": "unknown op"})
                continue
            if op != "delete" and not content:
                skipped.append({"op": op, "reason": "empty content"})
                continue
            if op != "append" and not target:
                skipped.append({"op": op, "reason": "missing target"})
                continue
            if op == "append" and _contains(text, content):
                skipped.append({"op": op, "reason": "duplicate"})
                continue

            if op == "append":
                candidate = (text + "\n" + content).strip()
            elif op == "insert_after":
                index = text.find(target)
                if index < 0:
                    skipped.append({"op": op, "reason": "target not found"})
                    continue
                insert_at = index + len(target)
                candidate = (text[:insert_at] + "\n" + content + text[insert_at:]).strip()
            elif op == "replace":
                if target not in text:
                    skipped.append({"op": op, "reason": "target not found"})
                    continue
                candidate = text.replace(target, content, 1).strip()
            else:  # delete
                if target not in text:
                    skipped.append({"op": op, "reason": "target not found"})
                    continue
                candidate = text.replace(target, "", 1).strip()

            if len(candidate) > max_chars:
                skipped.append({"op": op, "reason": "max_chars exceeded"})
                continue

            text = candidate
            applied.append({
                "op": op,
                "content": content,
                "target": target,
                "source_type": edit.get("source_type", "failure"),
            })

    return UpdateResult(content=text, applied=applied, skipped=skipped)


__all__ = ["DEFAULT_MAX_CHARS", "UpdateResult", "apply_patches"]
