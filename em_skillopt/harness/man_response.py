"""MAN request-local output repair, matching the pinned official Qwen path.

Infrastructure retries belong to the backend. These three attempts address
model output only, within one planner step, before executing any action.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from ..core.types import TokenUsage


@dataclass
class ManResponse:
    response: object
    plan: object
    usage: TokenUsage


def request_man_plan(llm, messages, parser, *, max_tokens, temperature, response_format):
    history = copy.deepcopy(messages)
    usage = TokenUsage()
    token_limit = max_tokens
    token_ceiling = max_tokens * 4
    for attempt in range(3):
        response = llm.generate(
            history, max_tokens=token_limit, temperature=temperature,
            response_format=response_format, stage="target",
        )
        usage = usage + response.usage
        finish = getattr(response, "finish_reason", "stop")
        if finish in {"content_filter", "refusal"}:
            raise RuntimeError(f"MAN provider refused output: {finish}")
        if finish == "length" or not response.text.strip():
            if attempt == 2:
                raise RuntimeError(f"MAN model returned {finish or 'empty content'} after 3 attempts")
            if finish == "length":
                if token_limit >= token_ceiling:
                    raise RuntimeError(f"MAN output reached token ceiling {token_ceiling}")
                token_limit = min(token_limit * 2, token_ceiling)
            continue
        try:
            # The official MAN decoder uses json.loads, not a separate schema
            # acceptance check. Keep tolerances in the semantic decoder.
            import json
            document = json.loads(response.text)
            plan = parser.parse_document(document, [])
            error = plan.error
        except (ValueError, TypeError, AttributeError) as exc:
            error = str(exc)
            plan = None
        if not error:
            plan.raw = response.text
            return ManResponse(response, plan, usage)
        if attempt == 2:
            raise RuntimeError(f"Invalid MAN output after 3 attempts: {error}")
        history.extend([
            {"role": "assistant", "content": response.text},
            {"role": "user", "content": "Your action plan cannot be parsed: " + error +
             ". Regenerate the complete JSON response. Every action must have seven discrete numeric values "
             "[x,y,z,rx,ry,rz,gripper], using either a JSON array or its quoted list representation. "
             "Use the provided object coordinates and rotation examples; no function calls, variable names, or prose. "
             "Position range 0..100, rotation range 0..119, gripper 0 or 1."},
        ])
    raise AssertionError("unreachable")
