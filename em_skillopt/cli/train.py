"""Minimal Method A training loop.

One epoch visits the train split once.  Episodes are grouped by capability and
split into rollout batches.  Each rollout batch is reflected in one or more
small minibatches, then update -> gate is run once per rollout batch.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..backends.qwen_target import QwenTargetClient
from ..core.skill import SkillStore
from ..data.splits import load_splits
from ..harness.parsing import ActionIdPlanParser, ManipulationPlanParser
from ..harness.prompts import build_context, response_format_for
from ..harness.runner import EpisodeRunner, RunnerConfig
from ..skillopt.aggregate import merge_patches
from ..skillopt.clip import rank_and_clip
from ..skillopt.gate import gate_candidate
from ..skillopt.reflect import ReflectConfig, reflect_episodes
from ..skillopt.rollout import RolloutConfig, rollout_specs
from ..skillopt.update import apply_patches
from .evaluate import build_adapter, build_env

os.environ.setdefault("DISPLAY", ":1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EM Method A training")
    parser.add_argument("--env", default="eb-nav", choices=["eb-nav", "eb-alf", "eb-hab", "eb-man"])
    parser.add_argument("--split-dir", default="splits/eb-nav")
    parser.add_argument("--out", default="")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--reflection-minibatch-size", type=int, default=8)
    parser.add_argument(
        "--reflection-workers",
        type=int,
        default=0,
        help="parallel reflection calls; 0 means number of minibatches",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="episode-level workers; 0 means use rollout batch size",
    )
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-planner-steps", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-skill-chars", type=int, default=6000)
    parser.add_argument("--max-edits", type=int, default=3)
    parser.add_argument("--min-reward-delta", type=float, default=0.0)
    parser.add_argument(
        "--gate-metric",
        default="hard",
        choices=["hard", "soft", "mixed"],
    )
    parser.add_argument("--gate-mixed-weight", type=float, default=0.5)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--n-shots", type=int, default=-1)
    parser.add_argument("--no-detection", action="store_true")
    parser.add_argument("--language-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _slice(items, limit: int):
    if limit and limit > 0:
        return items[:limit]
    return items


def _chunks(items, size: int):
    size = max(1, size)
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _step_buffer_text(entries: list[str], limit: int = 12) -> str:
    """Format previous failure patterns / rejected edits for reflect."""
    recent = entries[-limit:]
    if not recent:
        return ""
    return "\n".join(f"- {entry}" for entry in recent)


def _make_batches(specs, batch_size: int, rng: random.Random):
    """Group by capability so one batch needs only one simulator env."""

    groups = defaultdict(list)
    for spec in specs:
        groups[spec.capability].append(spec)

    batches = []
    for capability in sorted(groups):
        group = list(groups[capability])
        rng.shuffle(group)
        batches.extend(list(_chunks(group, batch_size)))
    rng.shuffle(batches)
    return batches


def _make_runner_factory(args, out_dir: Path, image_root: Path, client):
    counter = itertools.count()

    def make_runner(capability: str) -> EpisodeRunner:
        # Parallel workers must not share image filenames.
        images = image_root / capability / f"runner_{next(counter):03d}"
        env = build_env(
            args.env,
            capability,
            out_dir,
            image_dir=images,
        )
        adapter = build_adapter(
            args.env,
            env,
            out_dir,
            args,
            image_dir=images,
        )
        context = build_context(
            args.env,
            env,
            eval_set=capability,
            n_shots=args.n_shots,
            language_only=args.language_only,
        )
        parser = (
            ManipulationPlanParser()
            if args.env == "eb-man"
            else ActionIdPlanParser()
        )
        return EpisodeRunner(
            env=adapter,
            context=context,
            parser=parser,
            llm=client,
            config=RunnerConfig(
                max_planner_steps=args.max_planner_steps,
                max_steps=(args.max_steps or None),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                response_format=response_format_for(
                    args.env,
                    language_only=args.language_only,
                ),
            ),
        )

    return make_runner


def main() -> None:
    args = parse_args()
    out = Path(
        args.out or f"runs/method_a/{args.env}"
    ).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_splits(args.split_dir)
    train_specs = _slice(list(bundle.train), args.train_limit)
    val_specs = _slice(list(bundle.val), args.val_limit)
    if not train_specs:
        raise RuntimeError("empty train split")
    if not val_specs:
        raise RuntimeError("empty val split")

    os.environ.setdefault("QWEN_CHAT_THINKING_MODE", "disabled")
    client = QwenTargetClient.from_env(model=args.model, proxy=args.proxy)
    store = SkillStore(out / "skill")
    if not store.current_path.exists():
        store.save("", version=0, promote_best=True)
    current = store.load_current()
    current_val_summary = None
    best_val_summary = None
    best_step = 0
    val_cache: dict = {}
    step_buffer: list[str] = []

    step = 0
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_accepted_steps = 0
            epoch_accepted_edits = 0
            epoch_rejected_edits = 0
            rng = random.Random(args.seed + epoch)
            batches = _make_batches(
                train_specs,
                args.rollout_batch_size,
                rng,
            )
            print(
                f"[train] epoch {epoch}: "
                f"{len(batches)} rollout batch(es), "
                f"{len(train_specs)} episode(s)"
            )

            for batch_specs in batches:
                step += 1
                step_dir = out / f"step_{step:04d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                print("\n" + "-" * 72)
                print(
                    f"[train] epoch {epoch} step {step} | "
                    f"rollout episodes={len(batch_specs)}"
                )
                factory = _make_runner_factory(
                    args,
                    step_dir,
                    step_dir / "images",
                    client,
                )

                workers = args.parallel_workers or len(batch_specs)
                train_results = rollout_specs(
                    batch_specs,
                    current.content or None,
                    out_dir=step_dir / "rollout",
                    runner_factory=factory,
                    config=RolloutConfig(
                        resume=False,
                        verbose=True,
                        workers=workers,
                    ),
                )
                if not train_results:
                    print("[train] no rollout results; skip update")
                    continue

                print(f"[train] step {step} | reflect")
                mb_size = args.reflection_minibatch_size
                mb_chunks = list(_chunks(train_results, mb_size))
                patches_by_index: dict[int, list[dict]] = {}

                def reflect_one(mb_index: int, mb):
                    return mb_index, reflect_episodes(
                        mb,
                        current.content or None,
                        llm=client,
                        out_dir=step_dir / "reflect" / f"mb_{mb_index:02d}",
                        config=ReflectConfig(max_edits=args.max_edits),
                        context=_step_buffer_text(step_buffer),
                    )

                reflect_workers = min(
                    args.reflection_workers or len(mb_chunks),
                    len(mb_chunks),
                )
                with ThreadPoolExecutor(max_workers=reflect_workers) as pool:
                    futures = [
                        pool.submit(reflect_one, index, mb)
                        for index, mb in enumerate(mb_chunks, start=1)
                    ]
                    for future in as_completed(futures):
                        try:
                            index, patch_list = future.result()
                            patches_by_index[index] = patch_list
                        except Exception as exc:
                            print(f"[train] reflection failed: {exc!r}")

                patches = []
                for index in sorted(patches_by_index):
                    patches.extend(patches_by_index[index])

                print(f"[train] step {step} | aggregate + clip")
                merged_patch = merge_patches(
                    patches,
                    llm=client,
                    current_skill=current.content,
                )
                selected_patch = rank_and_clip(
                    merged_patch,
                    max_edits=args.max_edits,
                    current_skill=current.content,
                    llm=client,
                )
                (step_dir / "merged_patch.json").write_text(
                    json.dumps(merged_patch, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (step_dir / "selected_patch.json").write_text(
                    json.dumps(selected_patch, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                update = apply_patches(
                    current.content,
                    [selected_patch],
                    max_chars=args.max_skill_chars,
                )
                (step_dir / "candidate.md").write_text(
                    update.content.rstrip() + "\n",
                    encoding="utf-8",
                )

                # Skip the expensive gate if reflection produced no change.
                if update.content.strip() == (current.content or "").strip():
                    print(f"[train] step {step} | unchanged; gate skipped")
                    gate_dict = {"accepted": False, "reason": "unchanged"}
                else:
                    print(f"[train] step {step} | gate on val")
                    gate_factory = _make_runner_factory(
                        args,
                        step_dir / "gate",
                        step_dir / "gate" / "images",
                        client,
                    )
                    gate = gate_candidate(
                        current.content or None,
                        update.content,
                        val_specs,
                        runner_factory=gate_factory,
                        out_dir=step_dir / "gate",
                        min_reward_delta=args.min_reward_delta,
                        workers=args.parallel_workers or len(val_specs),
                        current_summary=current_val_summary,
                        best_summary=best_val_summary,
                        best_step=best_step,
                        metric=args.gate_metric,
                        mixed_weight=args.gate_mixed_weight,
                        candidate_cache=val_cache,
                        step=step,
                    )
                    gate_dict = gate.to_dict()
                    if current_val_summary is None:
                        current_val_summary = gate.current
                    if best_val_summary is None:
                        best_val_summary = gate.best
                        best_step = gate.best_step
                    if gate.accepted and update.content.strip():
                        current = store.save(
                            update.content,
                            promote_best=(gate.action == "accept_new_best"),
                        )
                        current_val_summary = gate.candidate
                        if gate.action == "accept_new_best":
                            best_val_summary = gate.best
                            best_step = gate.best_step
                        print(
                            f"[train] accepted step {step} "
                            f"({gate.action}), version {current.version}"
                        )
                    else:
                        print(f"[train] rejected step {step}: {gate.reason}")

                merged_count = merged_patch["num_merged_edits"]
                selected_count = selected_patch["ranking_details"]["selected_edits"]
                applied_count = len(update.applied)
                accepted = bool(gate_dict.get("accepted", False))

                if accepted:
                    accepted_edits = applied_count
                    rejected_edits = selected_count - applied_count
                    epoch_accepted_steps += 1
                    epoch_accepted_edits += accepted_edits
                else:
                    accepted_edits = 0
                    rejected_edits = selected_count
                epoch_rejected_edits += rejected_edits

                print(
                    f"[train] step {step} result: "
                    f"merged_edits={merged_count} "
                    f"selected_edits={selected_count} "
                    f"accepted_edits={accepted_edits} "
                    f"rejected_edits={rejected_edits}"
                )

                record = {
                    "step": step,
                    "epoch": epoch,
                    "train_episodes": len(train_results),
                    "merged_edits": merged_count,
                    "selected_edits": selected_count,
                    "accepted_edits": accepted_edits,
                    "rejected_edits": rejected_edits,
                    "accepted": accepted,
                    "action": str(gate_dict.get("action", "reject")),
                    "reason": str(gate_dict.get("reason", "")),
                    "gate_metric": args.gate_metric,
                    "best_step": best_step,
                    "skill_version": current.version,
                }
                (step_dir / "step.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                # Step buffer: failure patterns and rejected edits are fed to
                # the next reflect call, matching SkillOpt's step context.
                for result in train_results:
                    if result.success:
                        continue
                    step_buffer.append(
                        f"step {step} failure "
                        f"{result.spec.capability}/{result.spec.episode_id}: "
                        f"{result.failure_reason or 'unknown'}"
                    )
                if not accepted:
                    for edit in selected_patch.get("edits", []):
                        content = str(edit.get("content", "")).strip()
                        if content:
                            step_buffer.append(
                                f"step {step} rejected edit: {content}"
                            )

            print("\n" + "=" * 72)
            print(
                f"[train] epoch {epoch} summary: "
                f"accepted_steps={epoch_accepted_steps} "
                f"accepted_edits={epoch_accepted_edits} "
                f"rejected_edits={epoch_rejected_edits}"
            )
            print("=" * 72)

        test_specs = _slice(list(bundle.test), args.test_limit)
        if test_specs:
            print(
                f"\n[train] final best-skill test | "
                f"episodes={len(test_specs)}"
            )
            best = store.load_best()
            test_dir = out / "test_best"
            test_factory = _make_runner_factory(
                args,
                test_dir,
                test_dir / "images",
                client,
            )
            rollout_specs(
                test_specs,
                best.content or None,
                out_dir=test_dir,
                runner_factory=test_factory,
                config=RolloutConfig(
                    resume=False,
                    verbose=True,
                    workers=args.parallel_workers or len(test_specs),
                ),
            )
        else:
            print("[train] no test split; skip final test")

        print(f"[train] done. best skill: {store.best_path}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
