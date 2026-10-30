"""Validate action semantics after JSON-schema checks.

JSON structure is validated in :mod:`harness.schema`.  This module only turns
the validated ``executable_plan`` into actions and checks environment-specific
constraints such as action-id ranges and 7D manipulation bounds.
"""
from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Sequence

from ..core.types import ActionPlan, ActionValue


def load_json_value(raw: str) -> Any:
    """Extract the first JSON object from a (possibly fenced) response."""

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    if start < 0:
        raise ValueError("response contains no JSON object")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


def load_json_object(raw: str) -> dict[str, Any]:
    value = load_json_value(raw)
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    return value


def _plan_items(document: dict[str, Any]) -> list:
    plan = document.get("executable_plan")
    if not isinstance(plan, list):
        raise ValueError("executable_plan must be a list")
    return plan


class ActionIdPlanParser:
    """Action-id parser for Nav / ALFRED / Habitat."""

    def parse_document(
        self,
        document: dict[str, Any],
        action_space: Sequence[ActionValue],
    ) -> ActionPlan:
        try:
            plan = _plan_items(document)
        except Exception as exc:
            return ActionPlan(error=f"parse error: {exc}")

        if not plan:
            # Original EM: empty plan -> no environment step, replan.
            return ActionPlan(actions=[])

        actions: list[ActionValue] = []
        for index, item in enumerate(plan):
            action_id = item.get("action_id") if isinstance(item, dict) else None
            if not isinstance(action_id, int) or isinstance(action_id, bool):
                return ActionPlan(
                    actions=actions,
                    error=f"non-integer action_id at index {index}: {action_id!r}",
                )
            # Out-of-range ids are kept: the environment maps them to random
            # actions, exactly like the original EM evaluator.
            actions.append(action_id)

        return ActionPlan(actions=actions)



class ManipulationPlanParser:
    """Parser for EB-Manipulation 7D discrete actions."""

    @staticmethod
    def _action_values(item: Any) -> Any:
        if isinstance(item, dict):
            return item.get("action")
        return item

    def parse_document(
        self,
        document: dict[str, Any],
        action_space: Sequence[ActionValue],
    ) -> ActionPlan:
        del action_space
        try:
            # Match the pinned official parse_manipulation_plan decoder.
            plan = document.get(
                "executable_plan", document.get("properties", {}).get("executable_plan")
            )
            if isinstance(plan, str):
                plan = ast.literal_eval(plan)
            if not isinstance(plan, (list, tuple)):
                raise ValueError("executable_plan must be a list")
        except (ValueError, TypeError, AttributeError, SyntaxError) as exc:
            return ActionPlan(error=f"parse error: {exc}")

        if not plan:
            return ActionPlan(stop=True)

        actions: list[ActionValue] = []
        for index, item in enumerate(plan):
            if isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], dict):
                item = item[0]
            values = self._action_values(item)
            if isinstance(values, str):
                try:
                    values = ast.literal_eval(values)
                except (ValueError, SyntaxError):
                    return ActionPlan(
                        error=f"action {index} is not a literal list"
                    )

            if not isinstance(values, (list, tuple)) or len(values) != 7:
                return ActionPlan(
                    error=f"action {index} must be a 7-element list"
                )
            if any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                or int(value) != value
                for value in values
            ):
                return ActionPlan(
                    error=f"action {index} must contain integers"
                )

            values = [int(value) for value in values]
            if any(not 0 <= value <= 100 for value in values[:3]):
                return ActionPlan(
                    error=f"action {index} position out of [0, 100]"
                )
            if any(not 0 <= value < 120 for value in values[3:6]):
                return ActionPlan(
                    error=f"action {index} rotation out of [0, 119]"
                )
            if values[6] not in (0, 1):
                return ActionPlan(
                    error=f"action {index} gripper must be 0 or 1"
                )
            actions.append(values)

        return ActionPlan(actions=actions)



__all__ = [
    "ActionIdPlanParser",
    "ManipulationPlanParser",
    "load_json_object",
    "load_json_value",
]
