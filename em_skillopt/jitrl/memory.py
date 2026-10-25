"""Non-parametric cross-episode memory used by JitRL Method B.

Each executed step is stored as ``(state, action, discounted return)``.  At
test time the current state retrieves similar past steps; their action returns
are converted into action advantages and used by ``JitRLPlanAdjuster``.

The official JitRL uses OpenAI embeddings + FAISS plus LLM step scoring.  This
EM adaptation deliberately uses a dependency-free token Jaccard similarity and
the environment's terminal success signal, so it can run without additional
API calls or a vector database.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core.types import EpisodeResult, TrajectoryStep

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def action_key(action: Any) -> str:
    """Stable key for int, str, or 7-D list actions."""

    if isinstance(action, (list, tuple)):
        return json.dumps(list(action), ensure_ascii=False)
    return json.dumps(action, ensure_ascii=False)


def action_value(key: str) -> Any:
    return json.loads(key)


@dataclass(frozen=True)
class JitRLRecord:
    state: str
    action: str
    ret: float
    success: bool
    episode_id: str
    env_name: str
    capability: str
    step: int


class JitRLMemory:
    """Append-only JSONL memory with token-similarity retrieval."""

    def __init__(self, path: str | Path, *, gamma: float = 0.95) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.gamma = float(gamma)
        self._records: list[JitRLRecord] = []
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._records.append(JitRLRecord(**json.loads(line)))
                except (TypeError, ValueError, json.JSONDecodeError):
                    # A truncated line should not poison an otherwise useful
                    # online memory.
                    continue

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self.path.write_text("", encoding="utf-8")

    def add_episode(
        self,
        result: EpisodeResult,
        *,
        success_bonus: float = 1.0,
    ) -> int:
        """Store every step with its discounted return.

        Sparse-reward EM environments only reveal success at the end, so the
        terminal success bonus is propagated backwards with ``gamma``.
        """

        trajectory = list(result.trajectory)
        if not trajectory:
            return 0

        returns = [0.0] * len(trajectory)
        running = 0.0
        for index in range(len(trajectory) - 1, -1, -1):
            step = trajectory[index]
            running = float(step.reward or 0.0) + self.gamma * running
            if step.success:
                running += float(success_bonus)
            returns[index] = running

        new_records: list[JitRLRecord] = []
        for index, step in enumerate(trajectory):
            state = (step.state_text or result.spec.instruction or "").strip()
            if not state:
                continue
            new_records.append(
                JitRLRecord(
                    state=state,
                    action=action_key(step.action),
                    ret=returns[index],
                    success=bool(step.success),
                    episode_id=result.spec.episode_id,
                    env_name=result.spec.env_name,
                    capability=result.spec.capability,
                    step=index,
                )
            )

        if not new_records:
            return 0

        with self._lock:
            self._records.extend(new_records)
            with self.path.open("a", encoding="utf-8") as handle:
                for record in new_records:
                    handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return len(new_records)

    def retrieve(
        self,
        query_state: str,
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[tuple[float, JitRLRecord]]:
        query_tokens = _tokens(query_state)
        if not query_tokens:
            return []
        with self._lock:
            records = list(self._records)
        scored = [
            (_jaccard(query_tokens, _tokens(record.state)), record)
            for record in records
        ]
        scored = [
            (score, record)
            for score, record in scored
            if score >= min_similarity and score > 0.0
        ]
        scored.sort(key=lambda item: (item[0], item[1].ret), reverse=True)
        return scored[: max(1, top_k)]

    def action_advantages(
        self,
        query_state: str,
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> tuple[dict[str, float], dict[str, int], float]:
        """Return per-action advantage, support count, and baseline return."""

        retrieved = self.retrieve(
            query_state,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        if not retrieved:
            return {}, {}, 0.0

        per_action: dict[str, list[float]] = {}
        all_returns: list[float] = []
        for _, record in retrieved:
            per_action.setdefault(record.action, []).append(record.ret)
            all_returns.append(record.ret)

        baseline = sum(all_returns) / len(all_returns)
        averages = {
            action: sum(values) / len(values)
            for action, values in per_action.items()
        }
        support = {
            action: len(values)
            for action, values in per_action.items()
        }
        advantages = {
            action: averages[action] - baseline
            for action in averages
        }
        return advantages, support, baseline

    def records(self) -> list[JitRLRecord]:
        with self._lock:
            return list(self._records)


__all__ = [
    "JitRLMemory",
    "JitRLRecord",
    "action_key",
    "action_value",
]
