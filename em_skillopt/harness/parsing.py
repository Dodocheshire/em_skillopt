"""Parse raw LLM responses into executable action plans.

Two parsers are provided:

- ``ActionIdPlanParser``: EB-ALFRED / EB-Habitat / EB-Navigation, where
  ``executable_plan`` is a list of ``{"action_id": int, ...}``.
- ``ManipulationPlanParser``: EB-Manipulation, where each action is a
  7-dimensional integer vector.

The parsers do not execute anything and do not import EmbodiedBench.  They only
turn text into ``ActionPlan``.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Sequence

from ..core.types import ActionPlan, ActionValue


def load_json_object(raw: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""

    text = raw.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contains no JSON object")

    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    return value


def coerce_plan_list(value: Any) -> list:
    """Accept list, JSON string, or Python-literal string."""

    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError("executable_plan is not a list")


def _item_action_id(item: Any) -> Any:
    if isinstance(item, dict):
        return item.get("action_id", item.get("action"))
    return item


class ActionIdPlanParser:
    """Parser for high-level action-id environments."""

    def parse(
        self,
        raw: str,
        action_space: Sequence[ActionValue],
    ) -> ActionPlan:
        try:
            document = load_json_object(raw)
            plan = coerce_plan_list(document.get("executable_plan"))
        except Exception as exc:
            return ActionPlan(raw=raw, error=f"parse error: {exc}")

        if not plan:
            return ActionPlan(raw=raw, stop=True)

        actions: list[ActionValue] = []
        for index, item in enumerate(plan):
            action_id = _item_action_id(item)
            if (
                not isinstance(action_id, int)
                or isinstance(action_id, bool) # bool is subclass of int, should be excluded
                or not 0 <= action_id < len(action_space)
            ):
                return ActionPlan(
                    actions=actions,
                    raw=raw,
                    error=(
                        f"invalid action_id at index {index}: "
                        f"{action_id!r} (space size={len(action_space)})"
                    ),
                )

            actions.append(action_id)

        return ActionPlan(actions=actions, raw=raw)


class ManipulationPlanParser:
    """Parser for EB-Manipulation 7D discrete actions."""

    def parse(
        self,
        raw: str,
        action_space: Sequence[ActionValue],
    ) -> ActionPlan:
        del action_space  # manipulation actions are not ids

        try:
            document = load_json_object(raw)
            plan = coerce_plan_list(document.get("executable_plan"))
        except Exception as exc:
            return ActionPlan(raw=raw, error=f"parse error: {exc}")

        if not plan:
            return ActionPlan(raw=raw, stop=True)

        actions: list[ActionValue] = []
        for index, item in enumerate(plan):
            if isinstance(item, dict):
                values = item.get("action")
            else:
                values = item

            if isinstance(values, str):
                try:
                    values = ast.literal_eval(values)
                except (ValueError, SyntaxError):
                    return ActionPlan(
                        raw=raw,
                        error=f"action {index} is not a literal list",
                    )

            if not isinstance(values, (list, tuple)) or len(values) != 7:
                return ActionPlan(
                    raw=raw,
                    error=f"action {index} must be a 7-element list",
                )

            if any( #好表达
                not isinstance(value, int) or isinstance(value, bool)
                for value in values
            ):
                return ActionPlan(
                    raw=raw,
                    error=f"action {index} must contain integers",
                )

            values = [int(value) for value in values]
            if any(not 0 <= value <= 100 for value in values[:3]):
                return ActionPlan(
                    raw=raw,
                    error=f"action {index} position out of [0, 100]",
                )
            if any(not 0 <= value < 120 for value in values[3:6]):
                return ActionPlan(
                    raw=raw,
                    error=f"action {index} rotation out of [0, 119]",
                )
            if values[6] not in (0, 1):
                return ActionPlan(
                    raw=raw,
                    error=f"action {index} gripper must be 0 or 1",
                )

            actions.append(values)

        return ActionPlan(actions=actions, raw=raw)


__all__ = [
    "ActionIdPlanParser",
    "ManipulationPlanParser",
    "coerce_plan_list",
    "load_json_object",
]