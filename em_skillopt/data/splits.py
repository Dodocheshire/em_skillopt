"""Simple episode-level random train/val split for Method A.

The official EM test split is kept untouched and is used for final evaluation.
Method A only needs a development train/val split; there is no dev test split.
Episodes are shuffled and split by count, without grouping by scene or
capability.  The layout is::

    <out>/
    ├── train/items.json
    ├── val/items.json
    └── split_manifest.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..core.types import EpisodeSpec

SPLIT_NAMES = ("train", "val")


@dataclass
class SplitBundle:
    train: list[EpisodeSpec] = field(default_factory=list)
    val: list[EpisodeSpec] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)


def build_splits(
    specs: Sequence[EpisodeSpec],
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> SplitBundle:
    """Shuffle episodes and make one train/val split.

    This deliberately does not group by scene or capability.  A fixed seed
    makes the split reproducible.
    """

    if not specs:
        raise ValueError("cannot split an empty episode list")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    env_names = {spec.env_name for spec in specs}
    if len(env_names) != 1:
        raise ValueError(
            "build_splits expects exactly one env_name; "
            f"got envs={sorted(env_names)}"
        )

    shuffled = list(specs)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    val_count = max(1, min(total - 1, round(total * val_ratio)))
    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]

    for spec in train_items:
        spec.split = "train"
    for spec in val_items:
        spec.split = "val"

    train_items.sort(key=lambda spec: spec.index)
    val_items.sort(key=lambda spec: spec.index)

    eval_sets = sorted({spec.capability for spec in specs})
    bundle = SplitBundle(
        train=train_items,
        val=val_items,
        manifest={
            "env_name": next(iter(env_names)),
            "eval_sets": eval_sets,
            "split_method": "episode_random",
            "seed": seed,
            "val_ratio": val_ratio,
            "counts": {
                "train": len(train_items),
                "val": len(val_items),
            },
            "capability_counts": {
                "train": dict(Counter(spec.capability for spec in train_items)),
                "val": dict(Counter(spec.capability for spec in val_items)),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return bundle


def save_splits(bundle: SplitBundle, out_dir: str | Path) -> Path:
    """Write train/val item lists and the split manifest."""

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
        manifest=manifest,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a simple random EM Method A train/val split"
    )
    parser.add_argument("--items", required=True, help="input EpisodeSpec JSON list")
    parser.add_argument("--out", required=True, help="output split directory")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    specs = load_items(args.items)
    bundle = build_splits(specs, val_ratio=args.val_ratio, seed=args.seed)
    root = save_splits(bundle, args.out)
    print(
        f"wrote {root}: "
        f"train={len(bundle.train)} "
        f"val={len(bundle.val)} "
        f"eval_sets={bundle.manifest['eval_sets']}"
    )


__all__ = [
    "SPLIT_NAMES",
    "SplitBundle",
    "build_splits",
    "save_splits",
    "load_items",
    "load_splits",
]


if __name__ == "__main__":
    main()
