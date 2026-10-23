#! WARNING Not Read
"""Run mixed-capability splits with one runner per capability.

A development split for EM can contain specs from several eval sets:

    train:
      eb-nav:base:0001
      eb-nav:common_sense:0003
      ...

One simulator instance is tied to one eval set (and therefore one
``capability``).  ``CapabilityBatchRunner`` groups specs by capability,
constructs one runner per group through a factory, and reuses the existing
``BatchRunner`` semantics (resume, artifacts, continue-on-error).

The target LLM client is intentionally not created or closed here; the caller
should share one client across all capabilities.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

from ..core.artifacts import RunArtifacts
from ..core.types import EpisodeResult, EpisodeSpec
from .batch import BatchConfig, BatchRunner
from .runner import EpisodeRunner


RunnerFactory = Callable[[str], EpisodeRunner]


class CapabilityBatchRunner:
    """Execute a mixed-capability spec list sequentially.

    Parameters
    ----------
    runner_factory:
        Called once per distinct ``spec.capability``.  It must build a runner
        whose adapter/environment is configured for that capability.
    artifacts:
        Optional shared artifact store.  Episode IDs are already unique across
        eval sets.
    config:
        Resume / continue-on-error policy, passed through to ``BatchRunner``.
    close_runners:
        Close each runner's ``env`` after its capability group finishes.
        The shared target LLM client is never closed.
    """

    def __init__(
        self,
        runner_factory: RunnerFactory,
        *,
        artifacts: RunArtifacts | None = None,
        config: BatchConfig | None = None,
        close_runners: bool = True,
    ) -> None:
        self.runner_factory = runner_factory
        self.artifacts = artifacts
        self.config = config or BatchConfig()
        self.close_runners = close_runners

    def run(
        self,
        specs: Sequence[EpisodeSpec],
        *,
        skill: str | None = None,
    ) -> list[EpisodeResult]:
        groups: dict[str, list[EpisodeSpec]] = defaultdict(list)
        for spec in specs:
            groups[spec.capability].append(spec)

        results: list[EpisodeResult] = []
        for capability in sorted(groups):
            group = groups[capability]
            print(
                f"[capability] {capability}: "
                f"{len(group)} episode(s)"
            )
            runner = self.runner_factory(capability)
            try:
                results.extend(
                    BatchRunner(
                        runner,
                        artifacts=self.artifacts,
                        config=self.config,
                    ).run(group, skill=skill)
                )
            finally:
                if self.close_runners:
                    self._close_runner(runner)

        return results

    @staticmethod
    def _close_runner(runner: EpisodeRunner) -> None:
        env = getattr(runner, "env", None)
        close = getattr(env, "close", None)
        if callable(close):
            close()


__all__ = ["CapabilityBatchRunner", "RunnerFactory"]
