"""Group-aware train/val/test splits for Method A.

EM-Benchmark has no official train split, so Method A must create development
splits without leaking the official test episodes.  Splits are made by
*group*, not by individual episode:

- EB-Navigation: scene
- EB-ALFRED: task family
- EB-Habitat: scene
- EB-Manipulation: task variation

All episodes in one group stay in exactly one split.  The output layout is::

    <out>/
    ├── train/items.json
    ├── val/items.json
    ├── test/items.json
    └── split_manifest.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from ..core.types import EpisodeSpec

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitBundle:
    train: list[EpisodeSpec] = field(default_factory=list)
    val: list[EpisodeSpec] = field(default_factory=list)
    test: list[EpisodeSpec] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)


def _norm_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("ratios must contain train, val, test")
    values = [float(item) for item in ratios]
    if any(item <= 0 for item in values):
        raise ValueError("all split ratios must be positive")
    total = sum(values)
    return tuple(item / total for item in values)  # type: ignore[return-value]


def _fallback_group_key(spec: EpisodeSpec) -> str:
    return f"{spec.env_name}:{spec.capability}"


def group_key(spec: EpisodeSpec) -> str:
    """Return the anti-leakage group for one episode.
    eval_set 是能力维度划分的子集，比如 base、common_sense(对应EpisodeSpec的capability)
    group 是这个数据集内部用来防泄漏的单位，比如 scene / task family
    例如，FloorPlan1 这个 scene 是一个 group，整个 group 只能进一个 split
    """

    meta = spec.metadata or {}

    if spec.env_name == "eb-nav":
        scene = meta.get("scene")
        return f"scene:{scene}" if scene else _fallback_group_key(spec)

    if spec.env_name == "eb-alf":
        task = str(meta.get("task") or "")
        family = task.split("/")[0] if task else ""
        return f"task:{family}" if family else _fallback_group_key(spec)

    if spec.env_name == "eb-hab":
        scene = meta.get("scene") or meta.get("scene_id")
        return f"scene:{scene}" if scene else _fallback_group_key(spec)

    if spec.env_name == "eb-man":
        variation = meta.get("task_variation")
        return (
            f"variation:{variation}"
            if variation
            else _fallback_group_key(spec)
        )

    return _fallback_group_key(spec)


def _assign_groups(
    groups: dict[str, list[EpisodeSpec]],
    ratios: tuple[float, float, float],
    rng: random.Random,
) -> dict[str, str]:
    """Assign whole groups to splits, balancing target episode counts."""

    keys = list(groups)
    rng.shuffle(keys)
    # Largest first gives better count balance; random order breaks ties.
    keys.sort(key=lambda key: len(groups[key]), reverse=True)

    total = sum(len(items) for items in groups.values())
    targets = [total * ratio for ratio in ratios]
    counts = [0, 0, 0]
    assignment: dict[str, str] = {}

    # 遍历每个分组, 例如scene:FloorPlan1、scene:FloorPlan2
    for key in keys:
        size = len(groups[key])
        # 在 train / val / test 里选一个 split：选差得最多的那个
        split_index = max(
            range(3),
            key=lambda idx: (targets[idx] - counts[idx], -idx), # 平局时优先选前面的
        )
        assignment[key] = SPLIT_NAMES[split_index]
        counts[split_index] += size

    return assignment


def build_splits(
    specs: Sequence[EpisodeSpec],
    *,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    min_groups: int = 3,
    group_key_fn: Callable[[EpisodeSpec], str] = group_key,
    allow_mixed_eval_sets: bool = False,
) -> SplitBundle:
    """Split specs by group into train/val/test.

    Raises:
        ValueError: if the input has too few groups to make three non-empty
            splits without putting one group in multiple splits.
    """

    if not specs:
        raise ValueError("cannot split an empty episode list")

    env_names = {spec.env_name for spec in specs}
    eval_sets = {spec.capability for spec in specs}
    if len(env_names) != 1:
        raise ValueError(
            "build_splits expects exactly one env_name; "
            f"got envs={sorted(env_names)}"
        )
    if len(eval_sets) != 1 and not allow_mixed_eval_sets:
        raise ValueError(
            "build_splits expects one eval_set unless "
            "allow_mixed_eval_sets=True; "
            f"got eval_sets={sorted(eval_sets)}"
        )

    normalized = _norm_ratios(ratios)
    groups: dict[str, list[EpisodeSpec]] = defaultdict(list)
    for spec in specs:
        groups[group_key_fn(spec)].append(spec)

    if len(groups) < min_groups:
        raise ValueError(
            f"need at least {min_groups} groups for train/val/test, "
            f"got {len(groups)}: {sorted(groups)}"
        )

    rng = random.Random(seed)
    assignment = _assign_groups(groups, normalized, rng)

    bundle = SplitBundle()
    for key, items in groups.items():
        split = assignment[key]
        for spec in items:
            spec.split = split
            getattr(bundle, split).append(spec)

    for name in SPLIT_NAMES:
        getattr(bundle, name).sort(key=lambda spec: spec.index)
        if not getattr(bundle, name):
            raise ValueError(
                f"split {name!r} is empty; adjust ratios or provide more groups"
            )

    group_to_split = {
        name: sorted(key for key, value in assignment.items() if value == name) # group key的list
        for name in SPLIT_NAMES
    }
    bundle.manifest = {
        "env_name": next(iter(env_names)),
        "eval_set": (
            next(iter(eval_sets)) if len(eval_sets) == 1 else "mixed"
        ),
        "eval_sets": sorted(eval_sets),
        "seed": seed,
        "ratios": dict(zip(SPLIT_NAMES, normalized)),
        "counts": {
            name: len(getattr(bundle, name)) for name in SPLIT_NAMES
        },
        "group_counts": {
            name: len(group_to_split[name]) for name in SPLIT_NAMES
        },
        "groups": group_to_split,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return bundle


def save_splits(bundle: SplitBundle, out_dir: str | Path) -> Path:
    """Write the standard split directory and manifest."""

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    for name in SPLIT_NAMES:
        split_dir = root / name
        split_dir.mkdir(parents=True, exist_ok=True)
        items = [asdict(spec) for spec in getattr(bundle, name)]
        (split_dir / "items.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (root / "split_manifest.json").write_text(
        json.dumps(bundle.manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root


def load_items(path: str | Path) -> list[EpisodeSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EpisodeSpec(**item) for item in data]


def load_splits(root: str | Path) -> SplitBundle:
    root_path = Path(root)
    manifest_path = root_path / "split_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    return SplitBundle(
        train=load_items(root_path / "train" / "items.json"),
        val=load_items(root_path / "val" / "items.json"),
        test=load_items(root_path / "test" / "items.json"),
        manifest=manifest,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped EM Method A splits from a specs JSON file"
    )
    parser.add_argument(
        "--items",
        required=True,
        help="input JSON list of EpisodeSpec dicts",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output split directory",
    )
    parser.add_argument("--ratios", default="0.7:0.15:0.15")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ratios = tuple(float(item) for item in args.ratios.split(":"))
    specs = load_items(args.items)
    bundle = build_splits(specs, ratios=ratios, seed=args.seed)
    root = save_splits(bundle, args.out)
    print(
        f"wrote {root}: "
        f"train={len(bundle.train)} "
        f"val={len(bundle.val)} "
        f"test={len(bundle.test)}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "SPLIT_NAMES",
    "SplitBundle",
    "build_splits",
    "group_key",
    "load_items",
    "load_splits",
    "save_splits",
]
