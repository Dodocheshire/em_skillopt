"""Strict JSON-schema validation for target-model responses."""
from __future__ import annotations

from typing import Any

from .parsing import load_json_object


def schema_body(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(response_format, dict):
        return None
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return None
    schema = json_schema.get("schema")
    return schema if isinstance(schema, dict) else None


def validate_json_schema(
    raw: str,
    response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    """Parse one JSON object and validate it against ``response_format``."""

    schema = schema_body(response_format)
    if schema is None:
        raise ValueError("no json_schema found in response_format")

    document = load_json_object(raw)

    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "jsonschema is required for strict response validation"
        ) from exc

    try:
        jsonschema.validate(instance=document, schema=schema)
    except jsonschema.ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path)
        location = path or "<root>"
        raise ValueError(
            f"schema validation failed at {location}: {exc.message}"
        ) from exc
    return document


__all__ = ["schema_body", "validate_json_schema"]
