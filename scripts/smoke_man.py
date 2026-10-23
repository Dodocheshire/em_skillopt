"""One-episode smoke test for EB-Manipulation.

Run in the embench_man conda env.  CoppeliaSim needs a display and the
library path from the EM installation:

    export COPPELIASIM_ROOT=/home/pengcheng/EM-Benchmark/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04
    export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY=:1
    PYTHONPATH=/home/pengcheng/EM-Benchmark/EmbodiedBench:/home/pengcheng/EM-Benchmark/em_skillopt \
      /opt/conda/envs/embench_man/bin/python scripts/smoke_man.py --model qwen3.8-27b
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("DISPLAY", ":1")

# EBManEnv uses the relative TTMS_FOLDER="embodiedbench/envs/eb_manipulation/".
# Run from the EmbodiedBench checkout so its data symlink resolves.
_EM_ROOT = Path("/home/pengcheng/EM-Benchmark/EmbodiedBench")
os.chdir(_EM_ROOT)

_COPPELIA = "/home/pengcheng/EM-Benchmark/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04"
os.environ.setdefault("COPPELIASIM_ROOT", _COPPELIA)
os.environ["LD_LIBRARY_PATH"] = (
    _COPPELIA + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", _COPPELIA)

from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
from embodiedbench.envs.eb_manipulation.eb_man_utils import (
    ROTATION_RESOLUTION,
    VOXEL_SIZE,
)
from embodiedbench.evaluator.config.system_prompts import (
    eb_manipulation_system_prompt,
)

from em_skillopt.backends.qwen_target import QwenTargetClient, QwenTargetConfig
from em_skillopt.core.types import EpisodeSpec, Observation, TrajectoryStep
from em_skillopt.harness.adapters.man import ManAdapter
from em_skillopt.harness.context import ContextBuilder
from em_skillopt.harness.parsing import ManipulationPlanParser
from em_skillopt.harness.runner import EpisodeRunner, RunnerConfig


def man_user_prompt(
    spec: EpisodeSpec,
    observation: Observation,
    history: list[TrajectoryStep],
    planner_step: int,
) -> str:
    coords = observation.metadata.get("object_coords", {})
    parts = [
        f"Human Instruction: {observation.text or spec.instruction}",
        f"Input: {coords}",
    ]
    if history:
        parts.append(
            "Previous actions: "
            + ", ".join(repr(step.action) for step in history[-5:])
        )
    parts.append(
        'Return ONLY JSON: {"executable_plan": "[[x,y,z,rx,ry,rz,gripper], ...]"}'
    )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("SMOKE_MODEL", "qwen3.8-27b"))
    parser.add_argument("--out", default="artifacts/smoke_man")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--no-detection", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = EBManEnv(
        eval_set="base",
        img_size=(500, 500),
        log_path=str(out / "images"),
        selected_indexes=[0],
    )
    adapter = ManAdapter(
        env,
        log_path=str(out / "images"),
        camera_views=["front_rgb"],
        detection_box=not args.no_detection,
        draw_coordinates=True,
        language_only=False,
        resolution=500,
    )

    spec = EpisodeSpec(
        episode_id="man:smoke:0000",
        env_name="eb-man",
        index=0,
        instruction="pick up the target object",
        capability="base",
        split="test",
        max_steps=args.max_steps,
    )

    observation = adapter.reset(spec)
    print("instruction:", observation.text)
    print("images:", observation.images)
    for path in observation.images:
        if not Path(path).exists():
            raise FileNotFoundError(f"missing image: {path}")
    print("object_coords:", observation.metadata.get("object_coords"))
    print("task_variation:", observation.metadata.get("task_variation"))

    base_system = eb_manipulation_system_prompt.format(
        VOXEL_SIZE,
        VOXEL_SIZE,
        int(360 / ROTATION_RESOLUTION),
        ROTATION_RESOLUTION,
        "",
    )
    context = ContextBuilder(
        base_system_prompt=base_system,
        user_builder=man_user_prompt,
    )

    baseline_messages = context.build(
        spec, observation, [], None, planner_step=0
    )
    skill = "- Move directly toward the target object before closing the gripper.\n"
    method_messages = context.build(
        spec, observation, [], skill, planner_step=0
    )

    print("baseline system chars:", len(baseline_messages[0]["content"]))
    print("method system chars:", len(method_messages[0]["content"]))
    print("skill injected:", "## Learned Skill (Method A)" in method_messages[0]["content"])
    print("baseline image parts:", baseline_messages[1]["content"][1:])
    print("method image parts:", method_messages[1]["content"][1:])

    client = QwenTargetClient(QwenTargetConfig.from_env(model=args.model))
    runner = EpisodeRunner(
        env=adapter,
        context=context,
        parser=ManipulationPlanParser(),
        llm=client,
        config=RunnerConfig(
            max_planner_steps=1,
            max_steps=args.max_steps,
            max_tokens=2048,
            temperature=0.0,
        ),
    )

    result = runner.run_episode(spec, skill=skill)
    print("episode success:", result.success)
    print("episode steps:", result.num_env_steps)
    print("episode failure_reason:", result.failure_reason)
    print("episode tokens:", vars(result.token_usage))
    if result.plans:
        print("last raw response:", result.plans[-1].raw_response[:800])

    (out / "result.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    client.close()
    env.close()


if __name__ == "__main__":
    main()
