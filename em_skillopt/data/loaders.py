"""Turn an already-constructed EM environment into EpisodeSpec objects.

The environment class owns dataset loading and index semantics.  This module
only translates those datasets into the harness contract, keeping the CLI free
of domain-specific dataset code.
"""
from __future__ import annotations

from typing import Any

from ..core.types import EpisodeSpec


def _max_steps(env: Any, default: int = 30) -> int:
    return int(getattr(env, "_max_episode_steps", default) or default)


def _nav_specs(env: Any, *, eval_set: str, split: str) -> list[EpisodeSpec]:
    specs: list[EpisodeSpec] = []
    for index, task in enumerate(env.dataset):
        specs.append(
            EpisodeSpec(
                episode_id=f"eb-nav:{eval_set}:{index:04d}",
                env_name="eb-nav",
                index=index,
                instruction=str(task.get("instruction", "")),
                capability=eval_set,
                split=split,
                max_steps=_max_steps(env, 20),
                metadata={
                    "eval_set": eval_set,
                    "scene": task.get("scene"),
                    "target": task.get("targetObjectType"),
                },
            )
        )
    return specs


def _alf_specs(env: Any, *, eval_set: str, split: str) -> list[EpisodeSpec]:
    specs: list[EpisodeSpec] = []
    for index, task in enumerate(env.dataset):
        specs.append(
            EpisodeSpec(
                episode_id=f"eb-alf:{eval_set}:{index:04d}",
                env_name="eb-alf",
                index=index,
                instruction=str(task.get("instruction", "")),
                capability=eval_set,
                split=split,
                max_steps=_max_steps(env, 30),
                metadata={
                    "eval_set": eval_set,
                    "task": task.get("task"),
                    "repeat_idx": task.get("repeat_idx"),
                },
            )
        )
    return specs


def _hab_specs(env: Any, *, eval_set: str, split: str) -> list[EpisodeSpec]:
    episodes = list(getattr(env.dataset, "episodes", []))
    specs: list[EpisodeSpec] = []
    for index, episode in enumerate(episodes):
        instruction = (
            getattr(episode, "instruction", "")
            or getattr(episode, "lang_goal", "")
            or ""
        )
        specs.append(
            EpisodeSpec(
                episode_id=f"eb-hab:{eval_set}:{index:04d}",
                env_name="eb-hab",
                index=index,
                instruction=str(instruction),
                capability=eval_set,
                split=split,
                max_steps=_max_steps(env, 30),
                metadata={"eval_set": eval_set},
            )
        )
    return specs


def _man_specs(env: Any, *, eval_set: str, split: str) -> list[EpisodeSpec]:
    specs: list[EpisodeSpec] = []
    for index, item in enumerate(env.dataset):
        variation = str(item[-1]) if item else ""
        specs.append(
            EpisodeSpec(
                episode_id=f"eb-man:{eval_set}:{index:04d}",
                env_name="eb-man",
                index=index,
                # EBManEnv.reset() returns the natural-language description;
                # ManAdapter fills Observation.text from it.
                instruction="",
                capability=eval_set,
                split=split,
                max_steps=_max_steps(env, 15),
                metadata={
                    "eval_set": eval_set,
                    "task_variation": variation,
                },
            )
        )
    return specs


def load_specs(
    env_name: str,
    env: Any,
    *,
    eval_set: str = "base",
    split: str = "test",
    limit: int = 0,
) -> list[EpisodeSpec]:
    """Build specs from an environment instance."""

    if env_name == "eb-nav":
        specs = _nav_specs(env, eval_set=eval_set, split=split)
    elif env_name == "eb-alf":
        specs = _alf_specs(env, eval_set=eval_set, split=split)
    elif env_name == "eb-hab":
        specs = _hab_specs(env, eval_set=eval_set, split=split)
    elif env_name == "eb-man":
        specs = _man_specs(env, eval_set=eval_set, split=split)
    else:
        raise ValueError(f"unknown env: {env_name}")

    if limit and limit > 0:
        specs = specs[:limit]
    return specs


__all__ = ["load_specs"]
