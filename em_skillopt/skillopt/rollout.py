"""Run one Method A rollout batch with a given skill.

This module is intentionally environment-agnostic.  The caller supplies a
``runner_factory(capability) -> EpisodeRunner`` that knows how to construct the
right simulator, adapter, prompt context, and parser.  The shared target LLM
client is owned by the caller.

Parallelism
-----------
When several workers are requested the specs are split into per-capability,
*contiguous, index-sorted* shards, and each shard is executed by one long-lived
runner.  Reusing a runner is important for simulators whose adapters can only
seek forward (e.g. Habitat, whose ``reset()`` advances an internal episode
iterator): a fresh env per episode would re-walk the whole dataset for every
episode, which dominates the runtime.

Set ``RolloutConfig.shard_capability = False`` to fall back to the legacy
one-env-per-episode behaviour.
"""
from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import threading
import time
from collections import defaultdict
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
    shard_capability: bool = True
    # Dropped episodes shrink the gate's denominator; retry before giving up.
    episode_retries: int = 2
    # Domains whose simulator has process-global state (habitat-sim, CoppeliaSim)
    # must never host two instances in one process; those run one env per
    # forked worker process instead of one per thread.
    process_isolation_envs: frozenset[str] = frozenset({"eb-hab", "eb-man"})


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
    proc_spec_factory=None,
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
            "shard_capability": cfg.shard_capability,
        },
    )

    needs_processes = (
        cfg.workers
        and cfg.workers > 1
        and specs[0].env_name in cfg.process_isolation_envs
    )
    if needs_processes:
        if proc_spec_factory is None:
            raise RuntimeError(
                f"{specs[0].env_name} keeps its simulator in process-global "
                "state; running it with threads produces corrupt observations. "
                "Pass proc_spec_factory so workers can be spawned."
            )
        results = _process_rollout(
            specs,
            skill,
            artifacts=artifacts,
            proc_spec_factory=proc_spec_factory,
            config=cfg,
        )
    elif cfg.workers and cfg.workers > 1:
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

    if specs[0].env_name == "eb-man" and not cfg.resume:
        expected = {spec.episode_id for spec in specs}
        actual = {result.spec.episode_id for result in results}
        if actual != expected:
            raise RuntimeError(
                f"Incomplete MAN batch; refusing update/gate: missing={sorted(expected-actual)}"
            )

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


def _sort_key(spec: EpisodeSpec):
    return (spec.index if spec.index is not None else 0, spec.episode_id)


def _make_shards(
    pending: Sequence[EpisodeSpec],
    workers: int,
) -> list[list[EpisodeSpec]]:
    """Split pending specs into contiguous, index-sorted per-capability shards.

    Shard count per capability is proportional to that capability's share of
    the batch, capped by ``workers`` and by the group size.
    """

    groups: dict[str, list[EpisodeSpec]] = defaultdict(list)
    for spec in pending:
        groups[spec.capability].append(spec)

    total = len(pending)
    shards: list[list[EpisodeSpec]] = []
    for capability in sorted(groups):
        group = sorted(groups[capability], key=_sort_key)
        share = workers * len(group) / total
        count = max(1, min(len(group), workers, int(round(share))))
        size = math.ceil(len(group) / count)
        for start in range(0, len(group), size):
            shards.append(group[start:start + size])
    return shards


def _parallel_rollout(
    specs: Sequence[EpisodeSpec],
    skill: str | None,
    *,
    artifacts: RunArtifacts,
    runner_factory: RunnerFactory,
    config: RolloutConfig,
) -> list[EpisodeResult]:
    """Run independent episodes concurrently.

    With ``shard_capability`` (default) each worker owns one simulator and walks
    a contiguous, index-ascending shard of a single capability.  Otherwise every
    episode gets a fresh env (legacy behaviour).
    """

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

    record_lock = threading.Lock()

    def record(result: EpisodeResult) -> None:
        # Save immediately: a long shard must not hold every episode in memory
        # until it ends, otherwise a crash loses the whole shard and there is no
        # progress signal while it runs.
        artifacts.save_episode(result)
        with record_lock:
            results_by_id[result.spec.episode_id] = result
            print(f"[parallel] done {result.spec.episode_id}", flush=True)

    workers = max(1, config.workers)

    if not config.shard_capability:
        def run_one(spec: EpisodeSpec) -> EpisodeResult:
            runner = runner_factory(spec.capability)
            try:
                return runner.run_episode(spec, skill=skill)
            finally:
                _close_runner(runner)

        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
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
                record(result)
    else:
        shards = _make_shards(pending, workers)
        print(
            f"[parallel] {len(pending)} episode(s) in "
            f"{len(shards)} shard(s), {min(workers, len(shards))} worker(s)"
        )

        def run_shard(shard: list[EpisodeSpec]) -> None:
            runner = runner_factory(shard[0].capability)
            try:
                for spec in shard:
                    result = None
                    for attempt in range(config.episode_retries + 1):
                        try:
                            result = runner.run_episode(spec, skill=skill)
                            break
                        except Exception as exc:  # noqa: BLE001
                            if attempt >= config.episode_retries:
                                print(
                                    f"[parallel] {spec.episode_id} failed "
                                    f"after {attempt + 1} attempt(s): {exc!r}",
                                    flush=True,
                                )
                                if not config.continue_on_error:
                                    raise
                            else:
                                print(
                                    f"[parallel] {spec.episode_id} retry "
                                    f"{attempt + 1}/{config.episode_retries}: "
                                    f"{exc!r}",
                                    flush=True,
                                )
                                time.sleep(min(30, 10 * (attempt + 1)))
                    if result is not None:
                        record(result)
            finally:
                _close_runner(runner)

        with ThreadPoolExecutor(
            max_workers=min(workers, len(shards))
        ) as pool:
            futures = [pool.submit(run_shard, shard) for shard in shards]
            for future in as_completed(futures):
                future.result()

    order = {spec.episode_id: index for index, spec in enumerate(specs)}
    return sorted(
        results_by_id.values(),
        key=lambda result: order[result.spec.episode_id],
    )


def _process_rollout(
    specs: Sequence[EpisodeSpec],
    skill: str | None,
    *,
    artifacts: RunArtifacts,
    proc_spec_factory,
    config: RolloutConfig,
) -> list[EpisodeResult]:
    """Sharded rollout with one simulator per spawned OS process."""

    pending = [
        spec for spec in specs
        if not (config.resume and artifacts.episode_done(spec.episode_id))
    ]
    if not pending:
        return []

    results_by_id: dict[str, EpisodeResult] = {}
    shards = _make_shards(pending, max(1, config.workers))

    ctx = mp.get_context("spawn")
    workers = min(len(shards), max(1, config.workers))
    print(
        f"[parallel] {len(pending)} episode(s) in {len(shards)} shard(s), "
        f"{workers} spawned process(es), one env per process",
        flush=True,
    )

    queue: list[list[EpisodeSpec]] = list(shards)
    running = []
    while queue or running:
        while queue and len(running) < workers:
            shard = queue.pop(0)
            runner_spec = proc_spec_factory(shard[0].capability)
            runner_spec.skill = skill
            runner_spec.out_dir = str(artifacts.root)
            runner_spec.image_dir = str(artifacts.root / "images" / shard[0].capability / f"shard_{shard[0].index}_{shard[-1].index}")
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            process = ctx.Process(
                target=_spawn_entry,
                args=(runner_spec, shard, child_conn),
                daemon=True,
            )
            process.start()
            child_conn.close()
            running.append((process, parent_conn, shard))

        process, conn, shard = running.pop(0)
        collected: list[EpisodeResult] = []
        while True:
            try:
                message = conn.recv()
            except EOFError:
                break
            kind = message[0]
            if kind == "ok":
                collected.extend(message[1])
            elif kind == "one":
                pass  # already persisted and announced by the child
            elif kind == "err":
                print(f"[parallel] {message[1]} failed: {message[2]}", flush=True)
            else:
                print(f"[parallel] shard fatal: {message[1]}", flush=True)
        process.join()
        conn.close()

        for result in collected:
            artifacts.save_episode(result)
            results_by_id[result.spec.episode_id] = result

    order = {spec.episode_id: index for index, spec in enumerate(specs)}
    return sorted(
        results_by_id.values(),
        key=lambda result: order[result.spec.episode_id],
    )


def _spawn_entry(runner_spec, shard, conn) -> None:
    """Top-level (picklable) entry point for a spawned worker."""

    from ..harness.procpool import run_shard

    run_shard(runner_spec, shard, conn)


def _close_runner(runner) -> None:
    env = getattr(runner, "env", None)
    close = getattr(env, "close", None)
    if callable(close):
        close()


__all__ = [
    "RolloutConfig",
    "rollout_specs",
]
