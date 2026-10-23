"""Baseline evaluation CLI for the four EM environments.

Examples::

    PYTHONPATH=/home/pengcheng/EM-Benchmark/EmbodiedBench:/home/pengcheng/EM-Benchmark/em_skillopt \
      /opt/conda/envs/embench_nav/bin/python -m em_skillopt.cli.evaluate \
      --env eb-nav --method baseline --limit 2 --model qwen3.8-27b
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("DISPLAY", ":1")

from ..backends.qwen_target import QwenTargetClient
from ..core.artifacts import RunArtifacts
from ..core.metrics import em_summary, format_summary, summarize
from ..harness.adapters.alf import AlfAdapter
from ..harness.adapters.hab import HabAdapter
from ..harness.adapters.man import ManAdapter
from ..harness.adapters.nav import NavAdapter
from ..harness.batch import BatchConfig, BatchRunner
from ..harness.parsing import ActionIdPlanParser, ManipulationPlanParser
from ..harness.prompts import build_context
from ..harness.runner import EpisodeRunner, RunnerConfig
from ..data.loaders import load_specs

EM_ROOT = Path("/home/pengcheng/EM-Benchmark/EmbodiedBench")
COPPELIA = (
    "/home/pengcheng/EM-Benchmark/"
    "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04"
)


def build_env(env_name: str, eval_set: str, out: Path) -> Any:
    """
    eval_set: EM 的能力子集（base / common_sense / complex_instruction / spatial / visual_appearance / long_horizon 等）
    """
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if env_name == "eb-nav":
        from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv
        return EBNavigationEnv(
            eval_set=eval_set,
            exp_name="em_skillopt_baseline",
            resolution=500,
            down_sample_ratio=1.0,
        )

    if env_name == "eb-alf":
        from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
        return EBAlfEnv(
            eval_set=eval_set,
            exp_name="em_skillopt_baseline",
            resolution=500,
            down_sample_ratio=1.0,
        )

    if env_name == "eb-hab":
        from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
        return EBHabEnv(
            eval_set=eval_set,
            exp_name="em_skillopt_baseline",
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


def build_adapter(env_name: str, env: Any, out: Path, args: argparse.Namespace) -> Any:
    log_path = str(out / "images")
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
    parser = argparse.ArgumentParser(description="EM baseline evaluation")
    parser.add_argument(
        "--env",
        required=True,
        choices=["eb-nav", "eb-alf", "eb-hab", "eb-man"],
    )
    parser.add_argument("--method", default="baseline", choices=["baseline"])
    parser.add_argument("--eval-set", default="base")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-planner-steps", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n-shots", type=int, default=-1)
    parser.add_argument("--no-detection", action="store_true")
    parser.add_argument("--language-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method != "baseline":
        raise NotImplementedError("only --method baseline is wired up")

    out = Path(
        args.out or f"artifacts/baseline_{args.env}"
    ).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    env = build_env(args.env, args.eval_set, out)
    adapter = build_adapter(args.env, env, out, args)
    context = build_context(
        args.env,
        env,
        eval_set=args.eval_set,
        n_shots=args.n_shots,
        language_only=args.language_only,
    )
    specs = load_specs(
        args.env,
        env,
        eval_set=args.eval_set,
        split=args.split,
        limit=args.limit,
    )
    if not specs:
        raise RuntimeError("no episodes loaded")

    parser = (
        ManipulationPlanParser()
        if args.env == "eb-man"
        else ActionIdPlanParser()
    )
    # Default to non-thinking for reproducible baseline numbers.  The caller
    # can override this by exporting QWEN_CHAT_THINKING_MODE explicitly.
    os.environ.setdefault("QWEN_CHAT_THINKING_MODE", "disabled")
    client = QwenTargetClient.from_env(
        model=args.model,
        proxy=args.proxy,
    )
    runner = EpisodeRunner(
        env=adapter,
        context=context,
        parser=parser,
        llm=client,
        config=RunnerConfig(
            max_planner_steps=args.max_planner_steps,
            max_steps=(args.max_steps or None),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
    )

    artifacts = RunArtifacts(out)
    artifacts.save_json("config.json", vars(args))
    results = BatchRunner(
        runner,
        artifacts=artifacts,
        config=BatchConfig(
            resume=not args.no_resume,
            continue_on_error=True,
        ),
    ).run(specs, skill=None)

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

    client.close()
    env.close()


if __name__ == "__main__":
    main()
