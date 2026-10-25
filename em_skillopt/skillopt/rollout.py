"""Run one Method A rollout batch with a given skill.

This module is intentionally environment-agnostic.  The caller supplies a
``runner_factory(capability) -> EpisodeRunner`` that knows how to construct the
right simulator, adapter, prompt context, and parser.  The shared target LLM
client is owned by the caller.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..core.artifacts import RunArtifacts
from ..core.metrics import format_summary, summarize
from ..core.types import EpisodeResult, EpisodeSpec
from ..harness.batch import BatchConfig
from ..harness.capability import CapabilityBatchRunner, RunnerFactory


@dataclass
class RolloutConfig:
    resume: bool = True
    continue_on_error: bool = True
    verbose: bool = True
    workers: int = 1


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
            "workers": cfg.workers,
        },
    )

    if cfg.workers and cfg.workers > 1:
        results = _parallel_rollout(
            specs,
            skill,
            artifacts=artifacts,
            runner_factory=runner_factory,
            config=cfg,
        )
    else:
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


def _parallel_rollout(
    specs: Sequence[EpisodeSpec],
    skill: str | None,
    *,
    artifacts: RunArtifacts,
    runner_factory: RunnerFactory,
    config: RolloutConfig,
) -> list[EpisodeResult]:
    """Run independent episodes concurrently; each worker owns one env."""

    pending = [
        spec for spec in specs
        if not (
            config.resume
            and artifacts.episode_done(spec.episode_id)
        )
    ]
    results_by_id: dict[str, EpisodeResult] = {}
    if not pending:
        return []

    def run_one(spec: EpisodeSpec) -> EpisodeResult:
        runner = runner_factory(spec.capability)
        try:
            return runner.run_episode(spec, skill=skill)
        finally:
            env = getattr(runner, "env", None)
            close = getattr(env, "close", None)
            if callable(close):
                close()

    workers = min(max(1, config.workers), len(pending))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, spec): spec
            for spec in pending
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[parallel] {spec.episode_id} failed: {exc!r}")
                if not config.continue_on_error:
                    raise
                continue
            artifacts.save_episode(result)
            results_by_id[spec.episode_id] = result
            print(f"[parallel] done {spec.episode_id}")

    order = {spec.episode_id: index for index, spec in enumerate(specs)}
    return sorted(
        results_by_id.values(),
        key=lambda result: order[result.spec.episode_id],
    )


__all__ = [
    "RolloutConfig",
    "rollout_specs",
]
