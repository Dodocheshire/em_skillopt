"""Normalize only MAN demonstration action syntax for the Qwen JSON schema.

All text outside executable_plan values is preserved byte for byte. The action
numbers and their order are unchanged. No language_plan or other prose is added.
"""
import ast
import json
import re

_PLAN_KEY = re.compile(r'"executable_plan"\s*:\s*')


def normalize_man_example(example: str) -> str:
    matches = list(_PLAN_KEY.finditer(example))
    decoder = json.JSONDecoder()
    text = example
    for match in reversed(matches):
        start = match.end()
        value, length = decoder.raw_decode(example[start:])
        if isinstance(value, str):
            try:
                plan = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                # One shipped demonstration has an extra grouping bracket.
                # Recover only explicit seven-integer vectors, accepting no
                # residual names, expressions, or other content.
                vector = re.compile(r'\[\s*-?\d+(?:\s*,\s*-?\d+){6}\s*\]')
                found = list(vector.finditer(value))
                remainder = vector.sub('', value)
                if not found or re.sub(r'[\[\],\s]', '', remainder):
                    raise ValueError('Unrecoverable MAN example action syntax')
                plan = [ast.literal_eval(m.group(0)) for m in found]
        else:
            plan = value
        if not isinstance(plan, (list, tuple)):
            raise ValueError('MAN example executable_plan must contain a list')
        normalized = []
        for item in plan:
            action = item.get('action') if isinstance(item, dict) else item
            if isinstance(action, str):
                action = ast.literal_eval(action)
            if not isinstance(action, (list, tuple)) or len(action) != 7:
                raise ValueError('MAN example action must contain seven values')
            if any(type(v) is not int for v in action):
                raise ValueError('MAN demonstration action values must be integers')
            normalized.append({'action': list(action)})
        text = text[:start] + json.dumps(normalized, ensure_ascii=False) + text[start+length:]
    return text


def normalize_man_examples(examples):
    return {key: [normalize_man_example(example) for example in values]
            for key, values in examples.items()}
