"""Learnable skill text and a small versioned file store.

Baseline uses ``skill=None`` and never touches this module.  Method A loads a
``SkillDocument``, injects its text into the target prompt, and saves a new
version whenever a candidate is accepted.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

SKILL_BLOCK_HEADER = "## Learned Skill (Method A)"

SKILL_PRECEDENCE_NOTE = (
    "The guidance below is learned strategy. The hard action-validity rules "
    "and environment constraints in this prompt always take precedence."
)


def _normalize(content: str) -> str:
    return content.strip()


def render_skill_block(content: str | None) -> str:
    """Return the only prompt section Method A adds.

    Empty or ``None`` content returns ``""`` so baseline prompts are
    byte-identical to the original EM harness prompt.
    """

    text = _normalize(content or "")
    if not text:
        return ""
    return (
        f"{SKILL_BLOCK_HEADER}\n\n"
        f"{SKILL_PRECEDENCE_NOTE}\n\n"
        f"{text}\n"
    )


@dataclass(frozen=True)
class SkillDocument:
    """One skill version: text plus an integer version number."""

    content: str = ""
    version: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.normalized

    @property
    def normalized(self) -> str:
        return _normalize(self.content)

    @property
    def hash(self) -> str:
        text = self.normalized
        if not text:
            return ""
        return hashlib.sha256((text + "\n").encode("utf-8")).hexdigest()

    def render_block(self) -> str:
        return render_skill_block(self.content)

    @classmethod
    def empty(cls) -> "SkillDocument":
        return cls()

    @classmethod
    def from_file(cls, path: str | Path) -> "SkillDocument":
        skill_path = Path(path)
        if not skill_path.exists():
            return cls()
        return cls(content=skill_path.read_text(encoding="utf-8"))


class SkillStore:
    """Plain-Markdown skill versions under one directory.

    Layout::

        skill/
        |-- current.md
        |-- best.md
        `-- versions/
            |-- v0000.md
            `-- v0001.md
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    @property
    def current_path(self) -> Path:
        return self.root / "current.md"

    @property
    def best_path(self) -> Path:
        return self.root / "best.md"

    def list_versions(self) -> list[int]:
        versions: list[int] = []
        for path in self.versions_dir.glob("v*.md"):
            try:
                versions.append(int(path.stem[1:]))
            except ValueError:
                continue
        return sorted(versions)

    def next_version(self) -> int:
        versions = self.list_versions()
        return (max(versions) + 1) if versions else 1

    def version_path(self, version: int) -> Path:
        return self.versions_dir / f"v{version:04d}.md"

    def load_current(self) -> SkillDocument:
        if not self.current_path.exists():
            return SkillDocument.empty()
        versions = self.list_versions()
        return SkillDocument(
            content=self.current_path.read_text(encoding="utf-8"),
            version=(versions[-1] if versions else 0),
        )

    def load_best(self, *, fallback_to_current: bool = True) -> SkillDocument:
        if self.best_path.exists():
            return SkillDocument(
                content=self.best_path.read_text(encoding="utf-8"),
            )
        if fallback_to_current:
            return self.load_current()
        return SkillDocument.empty()

    def load_version(self, version: int) -> SkillDocument:
        path = self.version_path(version)
        if not path.exists():
            return SkillDocument.empty()
        return SkillDocument(
            content=path.read_text(encoding="utf-8"),
            version=version,
        )

    def save(
        self,
        content: str,
        *,
        version: int | None = None,
        promote_best: bool = False,
    ) -> SkillDocument:
        """Write a new skill version and update ``current.md``."""

        text = content.rstrip()
        if text:
            text += "\n"

        version = version if version is not None else self.next_version()
        self.version_path(version).write_text(text, encoding="utf-8")
        self.current_path.write_text(text, encoding="utf-8")

        if promote_best:
            self.best_path.write_text(text, encoding="utf-8")

        return SkillDocument(content=text, version=version)

    def promote_best(self) -> None:
        if self.current_path.exists():
            self.best_path.write_text(
                self.current_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )


__all__ = [
    "SKILL_BLOCK_HEADER",
    "SKILL_PRECEDENCE_NOTE",
    "SkillDocument",
    "SkillStore",
    "render_skill_block",
]