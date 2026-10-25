"""Generic episode loop for EM-Benchmark.

The runner is domain-agnostic.  Domain-specific behavior lives in:

- ``EnvironmentAdapter``: reset/step/action space/replan policy
- ``ContextBuilder``: prompt construction
- ``PlanParser``: raw text -> ActionPlan
- ``LLMClient``: target model call
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..core.protocols import (
    ContextBuilder,
    EnvironmentAdapter,
    LLMClient,
    PlanAdjuster,
    PlanParser,
)
from ..core.types import (
    ActionPlan,
    EpisodeResult,
    EpisodeSpec,
    PlanRecord,
    TokenUsage,
    TrajectoryStep,
)
from .schema import validate_json_schema


@dataclass
class RunnerConfig:
    max_planner_steps: int = 20
    max_steps: int | None = None
    max_tokens: int = 8192
    temperature: float | None = None
    response_format: dict[str, Any] | None = None

class EpisodeRunner:
    """Run one episode end-to-end."""

    def __init__(
        self,
        env: EnvironmentAdapter,
        context: ContextBuilder,
        parser: PlanParser,
        llm: LLMClient,
        config: RunnerConfig | None = None,
        plan_adjuster: PlanAdjuster | None = None,
    ) -> None:
        self.env = env
        self.context = context
        self.parser = parser
        self.llm = llm
        self.plan_adjuster = plan_adjuster
        self.config = config or RunnerConfig()
        if not self.config.response_format:
            raise ValueError(
                "EpisodeRunner requires a JSON-schema response_format"
            )

    def run_episode(
        self,
        spec: EpisodeSpec,
        skill: str | None = None,
    ) -> EpisodeResult:
        started_at = time.perf_counter()

        max_steps = (
            self.config.max_steps
            or min(spec.max_steps, self.env.max_steps)
        )

        observation = self.env.reset(spec)
        history: list[TrajectoryStep] = []
        plans: list[PlanRecord] = []
        token_usage = TokenUsage()

        success = False
        done = False
        invalid_actions = 0
        format_errors = 0
        target_system_prompt = ""
        target_user_prompt = ""

        for planner_step in range(self.config.max_planner_steps):
            messages = self.context.build(
                spec,
                observation,
                history,
                skill,
                planner_step=planner_step,
            )
            if planner_step == 0:
                for message in messages:
                    content = message.get("content", "")
                    if isinstance(content, list):
                        text = "\n".join(
                            str(part.get("text", ""))
                            for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        )
                    else:
                        text = str(content)
                    if message.get("role") == "system":
                        target_system_prompt = text
                    elif message.get("role") == "user":
                        target_user_prompt = text

            response = self.llm.generate(
                messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format=self.config.response_format,
                stage="target",
            )
            token_usage = token_usage + response.usage

            try:
                document = validate_json_schema(
                    response.text,
                    self.config.response_format,
                )
            except Exception as exc:
                plan = ActionPlan(
                    raw=response.text,
                    error=f"schema validation error: {exc}",
                )
            else:
                plan = self.parser.parse_document(
                    document,
                    self.env.action_space,
                )
                plan.raw = response.text
                if (
                    self.plan_adjuster is not None
                    and plan.error is None
                    and plan.actions
                ):
                    plan = self.plan_adjuster.adjust(
                        plan,
                        spec,
                        observation,
                        history,
                        self.env.action_space,
                        planner_step,
                    )
            plans.append(
                PlanRecord(
                    planner_step=planner_step,
                    raw_response=response.text,
                    actions=plan.actions,
                    error=plan.error,
                )
            )

            if plan.error:
                format_errors += 1

            # 空计划、模型主动停止、解析失败且没有有效前缀，都结束当前 episode。
            if plan.stop or not plan.actions:
                break

            for action in plan.actions:
                if len(history) >= max_steps:
                    break

                state_text = self._state_text(spec, observation, history)
                step = self.env.step(action)
                observation = step.observation
                history.append(
                    TrajectoryStep(
                        planner_step=planner_step,
                        env_step=len(history) + 1,
                        action=action,
                        action_valid=step.action_valid,
                        reward=step.reward,
                        feedback=step.feedback,
                        done=step.done,
                        success=step.success,
                        state_text=state_text,
                    )
                )

                if not step.action_valid:
                    invalid_actions += 1 # 我记得invalid action过多，即使没有hit max_steps，也会终止episode，标记failure

                success = bool(step.success)
                if step.done: # complete the episode
                    done = True
                    break

                if not step.action_valid and self.env.replan_on_invalid:
                    break

            if done:
                break
            if len(history) >= max_steps:
                break

        elapsed_s = time.perf_counter() - started_at
        reward = (
            sum(step.reward for step in history) / len(history) # average reward in episode, useful for 'soft score'
            if history
            else 0.0
        )

        failure_reason = ""
        if not success:
            if any(plan.error for plan in plans):
                failure_reason = "format_error"
            elif any(not plan.actions and plan.error is None for plan in plans):
                failure_reason = "empty_plan"
            elif len(history) >= max_steps:
                failure_reason = "max_steps"
            elif invalid_actions:
                failure_reason = "invalid_action"
            else:
                failure_reason = "not_success"

        result = EpisodeResult(
            spec=spec,
            success=success,
            reward=reward,
            num_env_steps=len(history),
            num_planner_steps=len(plans),
            invalid_actions=invalid_actions,
            format_errors=format_errors,
            elapsed_s=elapsed_s,
            failure_reason=failure_reason,
            target_system_prompt=target_system_prompt,
            target_user_prompt=target_user_prompt,
            plans=plans,
            trajectory=history,
            token_usage=token_usage,
        )
        if self.plan_adjuster is not None:
            observe = getattr(self.plan_adjuster, "observe", None)
            if callable(observe):
                observe(result)
        return result

    @staticmethod
    def _state_text(
        spec: EpisodeSpec,
        observation: Any,
        history: list[TrajectoryStep],
    ) -> str:
        parts = [spec.instruction, getattr(observation, "text", "")]
        for step in history[-6:]:
            parts.append(
                f"{step.env_step}: action={step.action!r} "
                f"valid={step.action_valid} feedback={step.feedback}"
            )
        return "\n".join(str(part) for part in parts if str(part).strip())


__all__ = [
    "EpisodeRunner",
    "RunnerConfig",
]