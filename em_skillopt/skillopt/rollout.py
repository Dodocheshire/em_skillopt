"""Run one Method A rollout batch with a given skill.

This module is intentionally environment-agnostic.  The caller supplies a
``runner_factory(capability) -> EpisodeRunner`` that knows how to construct the
right simulator, adapter, prompt context, and parser.  The shared target LLM
client is owned by the caller.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..core.artifacts import RunArtifacts
from ..core.metrics import format_summary, summarize
from ..core.types import EpisodeResult, EpisodeSpec
from ..data.splits import load_splits
from ..harness.batch import BatchConfig
from ..harness.capability import CapabilityBatchRunner, RunnerFactory


@dataclass
class RolloutConfig:
    resume: bool = True
    continue_on_error: bool = True
    verbose: bool = True


def _skill_hash(skill: str | None) -> str:
    text = (skill or "").strip()
    if not text:
        return ""
    return hashlib.sha256((text + "\n").encode("utf-8")).hexdigest()


def rollout_specs(
    specs: Sequence[EpisodeSpec],
    skill: str | None,
    *,
    out_dir: str | Path,
    runner_factory: RunnerFactory,
    config: RolloutConfig | None = None,
) -> list[EpisodeResult]:
    """Run specs with one skill and save artifacts under ``out_dir``.

    The output directory is a step directory, for example::

        runs/method_a/eb-nav/step_0001/
        ├── config.json
        ├── episodes/
        ├── images/
        ├── rollouts.json
        └── summary.json
    """

    if not specs:
        raise ValueError("rollout requires at least one EpisodeSpec")

    cfg = config or RolloutConfig()
    output = Path(out_dir).expanduser().resolve()
    artifacts = RunArtifacts(output)

    artifacts.save_json(
        "rollout_config.json",
        {
            "env_name": specs[0].env_name,
            "split": specs[0].split,
            "num_specs": len(specs),
            "capabilities": sorted({spec.capability for spec in specs}),
            "skill_sha256": _skill_hash(skill),
            "resume": cfg.resume,
        },
    )

    results = CapabilityBatchRunner(
        runner_factory,
        artifacts=artifacts,
        config=BatchConfig(
            resume=cfg.resume,
            continue_on_error=cfg.continue_on_error,
        ),
    ).run(specs, skill=skill)

    rollouts = []
    for result in results:
        row = result.to_rollout_result()
        row["episode_json"] = str(
            artifacts.episode_path(result.spec.episode_id)
        )
        rollouts.append(row)

    artifacts.save_json("rollouts.json", {"rollouts": rollouts})

    summary = summarize(
        results,
        env_name=specs[0].env_name,
        split=specs[0].split,
    )
    artifacts.save_summary(summary)
    if cfg.verbose:
        print(format_summary(summary))

    return results


def rollout_split(
    split_dir: str | Path,
    split_name: str,
    skill: str | None,
    *,
    out_dir: str | Path,
    runner_factory: RunnerFactory,
    limit: int = 0,
    config: RolloutConfig | None = None,
) -> list[EpisodeResult]:
    """Load a saved split and run one rollout batch."""

    bundle = load_splits(split_dir)
    specs = list(getattr(bundle, split_name))
    if limit and limit > 0:
        specs = specs[:limit]
    if not specs:
        raise ValueError(f"no episodes in {split_dir}/{split_name}")
    return rollout_specs(
        specs,
        skill,
        out_dir=out_dir,
        runner_factory=runner_factory,
        config=config,
    )


__all__ = [
    "RolloutConfig",
    "rollout_specs",
    "rollout_split",
]
