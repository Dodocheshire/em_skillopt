"""Evaluate baseline or a learned skill on EM.

Two modes:

1. Official single-capability mode: ``--eval-set base``
2. Mixed-capability split mode: ``--split-dir splits/eb-nav --split-name test``

In split mode, specs from several eval sets are grouped by capability and
executed through parallel episode workers.
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("DISPLAY", ":1")

from ..backends.qwen_target import QwenTargetClient
from ..core.artifacts import RunArtifacts
from ..core.metrics import em_summary, format_summary, summarize
from ..core.skill import SkillDocument
from ..data.loaders import load_specs
from ..data.splits import load_splits
from ..harness.adapters.alf import AlfAdapter
from ..harness.adapters.hab import HabAdapter
from ..harness.adapters.man import ManAdapter
from ..harness.adapters.nav import NavAdapter
from ..harness.parsing import ActionIdPlanParser, ManipulationPlanParser
from ..harness.prompts import build_context, response_format_for
from ..harness.runner import EpisodeRunner, RunnerConfig
from ..skillopt.rollout import RolloutConfig, rollout_specs

EM_ROOT = Path("/home/pengcheng/EM-Benchmark/EmbodiedBench")
COPPELIA = (
    "/home/pengcheng/EM-Benchmark/"
    "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04"
)


def build_env(
    env_name: str,
    eval_set: str,
    out: Path,
    *,
    image_dir: Path | None = None,
) -> Any:
    """Build one env for one capability/eval set."""

    image_dir = image_dir or (out / "images")
    image_dir.mkdir(parents=True, exist_ok=True)

    if env_name == "eb-nav":
        from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv
        return EBNavigationEnv(
            eval_set=eval_set,
            exp_name="em_skillopt_eval",
            resolution=500,
            down_sample_ratio=1.0,
        )

    if env_name == "eb-alf":
        from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
        return EBAlfEnv(
            eval_set=eval_set,
            exp_name="em_skillopt_eval",
            resolution=500,
            down_sample_ratio=1.0,
        )

    if env_name == "eb-hab":
        from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
        return EBHabEnv(
            eval_set=eval_set,
            exp_name="em_skillopt_eval",
            resolution=500,
            down_sample_ratio=1.0,
        )

    if env_name == "eb-man":
        os.chdir(EM_ROOT)
        os.environ.setdefault("COPPELIASIM_ROOT", COPPELIA)
        os.environ["LD_LIBRARY_PATH"] = (
            COPPELIA + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", COPPELIA)
        os.environ.setdefault("EB_DETECTION_DEVICE", "cpu")
        from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
        return EBManEnv(
            eval_set=eval_set,
            img_size=(500, 500),
            log_path=str(image_dir),
        )

    raise ValueError(f"unknown env: {env_name}")


def build_adapter(
    env_name: str,
    env: Any,
    out: Path,
    args: argparse.Namespace,
    *,
    image_dir: Path | None = None,
) -> Any:
    log_path = str(image_dir or (out / "images"))
    if env_name == "eb-nav":
        return NavAdapter(env, log_path=log_path)
    if env_name == "eb-alf":
        return AlfAdapter(env, log_path=log_path)
    if env_name == "eb-hab":
        return HabAdapter(env, log_path=log_path)
    if env_name == "eb-man":
        return ManAdapter(
            env,
            log_path=log_path,
            camera_views=["front_rgb"],
            detection_box=not args.no_detection,
            draw_coordinates=True,
            language_only=args.language_only,
            resolution=500,
        )
    raise ValueError(f"unknown env: {env_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EM baseline / Method A evaluation")
    parser.add_argument(
        "--env",
        required=True,
        choices=["eb-nav", "eb-alf", "eb-hab", "eb-man"],
    )
    parser.add_argument(
        "--method",
        default="baseline",
        choices=["baseline", "method_a"],
    )
    parser.add_argument(
        "--skill-file",
        default="",
        help="learned skill markdown, required for --method method_a",
    )
    parser.add_argument("--eval-set", default="base")
    parser.add_argument(
        "--split-dir",
        default="",
        help="mixed-capability split root, e.g. splits/eb-nav",
    )
    parser.add_argument(
        "--split-name",
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="parallel episode workers (1 = sequential)",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-planner-steps", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n-shots", type=int, default=-1)
    parser.add_argument("--no-detection", action="store_true")
    parser.add_argument("--language-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def resolve_skill(args: argparse.Namespace) -> str | None:
    if args.method == "baseline":
        if args.skill_file:
            raise ValueError("--skill-file is only valid with --method method_a")
        return None

    if not args.skill_file:
        raise ValueError("--method method_a requires --skill-file")
    document = SkillDocument.from_file(args.skill_file)
    if document.is_empty:
        raise ValueError(f"skill file is empty: {args.skill_file}")
    return document.normalized


def make_runner_config(args: argparse.Namespace) -> RunnerConfig:
    return RunnerConfig(
        max_planner_steps=args.max_planner_steps,
        max_steps=(args.max_steps or None),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        response_format=response_format_for(
            args.env,
            language_only=args.language_only,
        ),
    )


def make_parser(env_name: str):
    return (
        ManipulationPlanParser()
        if env_name == "eb-man"
        else ActionIdPlanParser()
    )


def make_runner_factory(args, out: Path, client):
    counter = itertools.count()

    def make_runner(capability: str) -> EpisodeRunner:
        cap_images = out / "images" / capability / f"runner_{next(counter):03d}"
        env = build_env(
            args.env,
            capability,
            out,
            image_dir=cap_images,
        )
        adapter = build_adapter(
            args.env,
            env,
            out,
            args,
            image_dir=cap_images,
        )
        context = build_context(
            args.env,
            env,
            eval_set=capability,
            n_shots=args.n_shots,
            language_only=args.language_only,
        )
        return EpisodeRunner(
            env=adapter,
            context=context,
            parser=make_parser(args.env),
            llm=client,
            config=make_runner_config(args),
        )

    return make_runner


def run_single_capability(
    args: argparse.Namespace,
    out: Path,
    skill: str | None,
    client: QwenTargetClient,
) -> None:
    env = build_env(args.env, args.eval_set, out)
    try:
        specs = load_specs(
            args.env,
            env,
            eval_set=args.eval_set,
            split=args.split,
            limit=args.limit,
        )
    finally:
        env.close()
    if not specs:
        raise RuntimeError("no episodes loaded")

    artifacts = RunArtifacts(out)
    artifacts.save_json("config.json", vars(args))
    results = rollout_specs(
        specs,
        skill,
        out_dir=out,
        runner_factory=make_runner_factory(args, out, client),
        config=RolloutConfig(
            resume=not args.no_resume,
            verbose=False,
            workers=max(1, args.parallel_workers),
        ),
    )

    summary = summarize(results, env_name=args.env, split=args.split)
    artifacts.save_summary(summary)
    artifacts.save_json(
        "em_summary.json",
        em_summary(
            results,
            env_name=args.env,
            split=args.split,
            save_path=str(out),
        ),
    )
    print(format_summary(summary))


def run_split_dir(
    args: argparse.Namespace,
    out: Path,
    skill: str | None,
    client: QwenTargetClient,
) -> None:
    bundle = load_splits(args.split_dir)
    if bundle.manifest.get("env_name") not in ("", None, args.env):
        raise ValueError(
            f"split belongs to {bundle.manifest.get('env_name')}, "
            f"not {args.env}"
        )

    specs = list(getattr(bundle, args.split_name))
    if args.limit and args.limit > 0:
        specs = specs[: args.limit]
    if not specs:
        raise RuntimeError(f"no episodes in {args.split_dir}/{args.split_name}")

    artifacts = RunArtifacts(out)
    artifacts.save_json("config.json", vars(args))
    results = rollout_specs(
        specs,
        skill,
        out_dir=out,
        runner_factory=make_runner_factory(args, out, client),
        config=RolloutConfig(
            resume=not args.no_resume,
            verbose=False,
            workers=max(1, args.parallel_workers),
        ),
    )

    split_label = f"dev_{args.split_name}" # development split, not official test
    summary = summarize(results, env_name=args.env, split=split_label)
    artifacts.save_summary(summary)
    artifacts.save_json(
        "em_summary.json",
        em_summary(
            results,
            env_name=args.env,
            split=split_label,
            save_path=str(out),
        ),
    )
    print(format_summary(summary))


def main() -> None:
    args = parse_args()
    if args.method not in ("baseline", "method_a"):
        raise NotImplementedError(args.method)

    default_name = (
        f"artifacts/{args.method}_{args.env}"
        if not args.split_dir
        else f"artifacts/{args.method}_{args.env}_{args.split_name}"
    )
    out = Path(args.out or default_name).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    skill = resolve_skill(args)
    os.environ.setdefault("QWEN_CHAT_THINKING_MODE", "disabled")
    client = QwenTargetClient.from_env(model=args.model, proxy=args.proxy)
    try:
        if args.split_dir:
            run_split_dir(args, out, skill, client)
        else:
            run_single_capability(args, out, skill, client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
