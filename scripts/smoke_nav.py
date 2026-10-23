"""One-episode smoke test for EB-Navigation.

Run in the embench_nav conda env:

    DISPLAY=:1 PYTHONPATH=/home/pengcheng/EM-Benchmark/EmbodiedBench:/home/pengcheng/EM-Benchmark/em_skillopt \
      /opt/conda/envs/embench_nav/bin/python scripts/smoke_nav.py --model qwen3.8-27b
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("DISPLAY", ":1")

from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv
from embodiedbench.evaluator.config.eb_navigation_example import examples
from embodiedbench.evaluator.config.system_prompts import eb_navigation_system_prompt

from em_skillopt.backends.qwen_target import QwenTargetClient, QwenTargetConfig
from em_skillopt.core.types import EpisodeSpec
from em_skillopt.harness.adapters.nav import NavAdapter
from em_skillopt.harness.context import ContextBuilder, SimpleUserPromptBuilder
from em_skillopt.harness.parsing import ActionIdPlanParser
from em_skillopt.harness.runner import EpisodeRunner, RunnerConfig


def action_space_text(actions: list[str]) -> str:
    return ", ".join(
        f"action id {index}: {name}" for index, name in enumerate(actions)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("SMOKE_MODEL", "qwen3.8-27b"))
    parser.add_argument("--out", default="artifacts/smoke_nav")
    parser.add_argument("--max-steps", type=int, default=2)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = EBNavigationEnv(
        eval_set="base",
        exp_name="smoke_nav",
        resolution=500,
        down_sample_ratio=1.0,
    )
    adapter = NavAdapter(env, log_path=str(out / "images"))

    task = env.dataset[0]
    spec = EpisodeSpec(
        episode_id="nav:smoke:0000",
        env_name="eb-nav",
        index=0,
        instruction=str(task["instruction"]),
        capability="base",
        split="test",
        max_steps=args.max_steps,
    )

    observation = adapter.reset(spec)
    print("instruction:", spec.instruction)
    print("images:", observation.images)
    for path in observation.images:
        if not Path(path).exists():
            raise FileNotFoundError(f"missing image: {path}")

    base_system = eb_navigation_system_prompt.format(
        len(adapter.action_space) - 1,
        action_space_text(list(adapter.action_space)),
        "\n\n".join(str(item) for item in examples[:3]),
    )
    context = ContextBuilder(
        base_system_prompt=base_system,
        user_builder=SimpleUserPromptBuilder(
            first_step_suffix="Plan the next actions and output JSON.",
            replan_suffix="The previous plan did not finish. Replan and output JSON.",
        ),
    )

    baseline_messages = context.build(
        spec, observation, [], None, planner_step=0
    )
    skill = "- Prefer moving forward and avoid unnecessary rotation.\n"
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
        parser=ActionIdPlanParser(),
        llm=client,
        config=RunnerConfig(
            max_planner_steps=2,
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
