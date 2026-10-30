"""Rebuild a runner inside a spawned worker process.

habitat-sim and CoppeliaSim are process-global C/C++ simulators: two instances
in one process corrupt each other's observations (verified on hab: the same
episode rendered a different frame with 2 thread workers and a byte-identical
frame with 1). They therefore need one process per simulator.

``fork`` is not enough: when the parent has already initialised EGL (the
evaluator builds an env to enumerate the split), forked children die with
"unable to find CUDA device 0". So workers are *spawned* and everything the
runner needs must be expressible as plain data — that is what ``RunnerSpec``
is for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any


@dataclass
class RunnerSpec:
    """Everything needed to rebuild one runner in a fresh process."""

    env_name: str
    capability: str
    out_dir: str
    image_dir: str
    args: dict = field(default_factory=dict)
    skill: str | None = None

    @classmethod
    def build(
        cls,
        env_name: str,
        capability: str,
        out_dir,
        image_dir,
        args_namespace,
        skill: str | None = None,
    ) -> "RunnerSpec":
        return cls(
            env_name=env_name,
            capability=capability,
            out_dir=str(out_dir),
            image_dir=str(image_dir),
            args={
                key: value
                for key, value in vars(args_namespace).items()
                if isinstance(value, (str, int, float, bool, type(None), list, tuple))
            },
            skill=skill,
        )


def build_runner(spec: RunnerSpec):
    """Construct an ``EpisodeRunner`` from a ``RunnerSpec`` (child side)."""

    from ..backends.qwen_target import QwenTargetClient
    from ..cli.evaluate import build_adapter, build_env
    from ..harness.parsing import ActionIdPlanParser, ManipulationPlanParser
    from ..harness.prompts import build_context, response_format_for
    from ..harness.runner import EpisodeRunner, RunnerConfig

    args = SimpleNamespace(**spec.args)
    out = Path(spec.out_dir).expanduser().resolve()
    image_dir = Path(spec.image_dir).expanduser().resolve()

    env = build_env(spec.env_name, spec.capability, out, image_dir=image_dir)
    adapter = build_adapter(spec.env_name, env, out, args, image_dir=image_dir)
    context = build_context(
        spec.env_name,
        env,
        eval_set=spec.capability,
        n_shots=getattr(args, "n_shots", -1),
        language_only=getattr(args, "language_only", False),
    )
    parser = (
        ManipulationPlanParser()
        if spec.env_name == "eb-man"
        else ActionIdPlanParser()
    )
    llm = QwenTargetClient.from_env(
        model=getattr(args, "model", "qwen3.8-27b"),
        proxy=getattr(args, "proxy", None) or None,
    )
    config = RunnerConfig(
        max_planner_steps=getattr(args, "max_planner_steps", 20),
        max_steps=(getattr(args, "max_steps", 0) or None),
        max_tokens=getattr(args, "max_tokens", 8192),
        temperature=getattr(args, "temperature", 0.0),
        response_format=response_format_for(
            spec.env_name,
            language_only=getattr(args, "language_only", False),
        ),
    )
    return EpisodeRunner(
        env=adapter,
        context=context,
        parser=parser,
        llm=llm,
        config=config,
    )


def run_shard(spec: RunnerSpec, shard: list, conn) -> None:
    """Worker entry point: run every episode of ``shard`` in this process."""

    runner = None
    try:
        runner = build_runner(spec)
        produced = []
        for episode in shard:
            try:
                result = runner.run_episode(episode, skill=spec.skill)
            except Exception as exc:  # noqa: BLE001
                conn.send(("err", episode.episode_id, repr(exc)))
                continue
            produced.append(result)
            # Persist and announce immediately: a shard of 15 episodes must not
            # hold everything in memory until it ends (a crash would lose it and
            # the stall watchdog would see a frozen log).
            try:
                from ..core.artifacts import RunArtifacts

                RunArtifacts(Path(spec.out_dir)).save_episode(result)
            except Exception:  # noqa: BLE001
                pass
            print(f"[parallel] done {episode.episode_id}", flush=True)
            conn.send(("one", episode.episode_id))
        conn.send(("ok", produced))
    except Exception as exc:  # noqa: BLE001
        conn.send(("fatal", repr(exc)))
    finally:
        if runner is not None:
            env = getattr(runner, "env", None)
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["RunnerSpec", "build_runner", "run_shard"]
