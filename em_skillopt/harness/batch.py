"""Sequential batch execution for EM episodes.

The first version deliberately stays sequential: many EmbodiedBench simulators
are single-instance and not thread-safe.  Parallelism belongs in a later
process-pool layer, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..core.artifacts import RunArtifacts
from ..core.types import EpisodeResult, EpisodeSpec
from .runner import EpisodeRunner


@dataclass
class BatchConfig:
    resume: bool = True
    continue_on_error: bool = True


class BatchRunner:
    """Run a list of episodes through one ``EpisodeRunner``."""

    def __init__(
        self,
        runner: EpisodeRunner,
        *,
        artifacts: RunArtifacts | None = None,
        config: BatchConfig | None = None,
    ) -> None:
        self.runner = runner
        self.artifacts = artifacts
        self.config = config or BatchConfig()

    def run(
        self,
        specs: Sequence[EpisodeSpec],
        *,
        skill: str | None = None,
    ) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []

        for index, spec in enumerate(specs, start=1):
            if (
                self.artifacts is not None
                and self.config.resume
                and self.artifacts.episode_done(spec.episode_id)
            ):
                print(
                    f"[batch] {index}/{len(specs)} skip "
                    f"{spec.episode_id} (done)"
                )
                continue

            print(
                f"[batch] {index}/{len(specs)} run "
                f"{spec.episode_id} ({spec.env_name}/{spec.capability})"
            )

            try:
                result = self.runner.run_episode(spec, skill=skill)
            except Exception as exc:
                print(f"[batch] {spec.episode_id} failed: {exc!r}")
                if self.config.continue_on_error:
                    continue
                raise

            if self.artifacts is not None:
                self.artifacts.save_episode(result)

            results.append(result)

        return results


def run_episodes(
    runner: EpisodeRunner,
    specs: Iterable[EpisodeSpec],
    *,
    skill: str | None = None,
    artifacts: RunArtifacts | None = None,
    resume: bool = True,
    continue_on_error: bool = True,
) -> list[EpisodeResult]:
    """Functional wrapper around :class:`BatchRunner`."""

    return BatchRunner(
        runner,
        artifacts=artifacts,
        config=BatchConfig(
            resume=resume,
            continue_on_error=continue_on_error,
        ),
    ).run(list(specs), skill=skill)


__all__ = [
    "BatchConfig",
    "BatchRunner",
    "run_episodes",
]