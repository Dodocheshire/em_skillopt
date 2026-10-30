"""Run directories and episode result files.

Layout::

    run/
    |-- config.json
    |-- summary.json
    `-- episodes/
        |-- test_0001.json
        `-- test_0002.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator, Mapping

from .types import (EpisodeResult, EvalSummary, EpisodeSpec, PlanRecord,
                    TrajectoryStep, TokenUsage)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(value: str) -> str:
    cleaned = _SAFE_NAME.sub("_", str(value)).strip("_")
    return cleaned


class RunArtifacts:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    def episode_path(self, episode_id: str) -> Path:
        return self.episodes_dir / f"{safe_name(episode_id)}.json"

    def episode_done(self, episode_id: str) -> bool:
        path = self.episode_path(episode_id)
        return path.exists() and path.stat().st_size > 0

    def load_episode(self, episode_id: str) -> EpisodeResult:
        data = json.loads(self.episode_path(episode_id).read_text(encoding="utf-8"))
        data["spec"] = EpisodeSpec(**data["spec"])
        if data["spec"].episode_id != episode_id:
            raise ValueError(f"Episode identity mismatch for {episode_id}")
        data["plans"] = [PlanRecord(**row) for row in data.get("plans", [])]
        data["trajectory"] = [TrajectoryStep(**row) for row in data.get("trajectory", [])]
        data["token_usage"] = TokenUsage(**data.get("token_usage", {}))
        return EpisodeResult(**data)

    def save_episode(self, result: EpisodeResult) -> Path:
        path = self.episode_path(result.spec.episode_id)
        path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    # yield turns a function into a generator. Instead of computing and returning everything at once, 
    # the function produces one value at a time and pauses after each one, resuming when the next value is requested.
    def iter_episode_dicts(self) -> Iterator[dict[str, Any]]:
        for path in sorted(self.episodes_dir.glob("*.json")):
            yield json.loads(path.read_text(encoding="utf-8"))

    def save_summary(self, summary: EvalSummary | Mapping[str, Any]) -> None:
        data = summary.to_dict() if hasattr(summary, "to_dict") else dict(summary)
        self.save_json("summary.json", data)

    def save_config(self, config: Mapping[str, Any]) -> Path:
        return self.save_json("config.json", config)

    def save_json(self, name: str, data: Mapping[str, Any]) -> Path:
        path = self.root / name
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def load_json(self, name: str) -> dict[str, Any]:
        path = self.root / name
        return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "RunArtifacts",
    "safe_name",
]