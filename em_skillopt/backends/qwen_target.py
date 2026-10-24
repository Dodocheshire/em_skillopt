"""Qwen multimodal target client.

This is the target-side LLM client used by ``EpisodeRunner``.  It converts local
``image_path`` content parts into OpenAI-style ``image_url`` data URLs and calls
a Qwen OpenAI-compatible endpoint.

The optimizer has its own text-only client; this module only handles target
calls.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import httpx
from openai import OpenAI

from ..core.types import LLMResponse, Message, TokenUsage


@dataclass
class QwenTargetConfig:
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout_s: float = 120.0
    max_retries: int = 3
    enable_thinking: bool | None = None
    response_format: dict[str, Any] | None = None
    proxy: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        proxy: str | None = None,
    ) -> "QwenTargetConfig":
        base_url = (
            os.getenv("QWEN_CHAT_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or ""
        ).strip()
        api_key = (
            os.getenv("QWEN_CHAT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
        model_name = (
            model
            or os.getenv("QWEN_CHAT_MODEL")
            or os.getenv("OPENAI_MODEL")
            or ""
        ).strip()

        if not base_url:
            raise ValueError(
                "QWEN_CHAT_BASE_URL / OPENAI_BASE_URL is not set"
            )
        if not api_key:
            raise ValueError(
                "QWEN_CHAT_API_KEY / OPENAI_API_KEY is not set"
            )
        if not model_name:
            raise ValueError("QWEN_CHAT_MODEL / OPENAI_MODEL is not set")

        thinking_raw = os.getenv(
            "QWEN_CHAT_THINKING_MODE", ""
        ).strip().lower()
        enable_thinking: bool | None
        if thinking_raw in {"enabled", "true", "1"}:
            enable_thinking = True
        elif thinking_raw in {"disabled", "false", "0"}:
            enable_thinking = False
        else:
            enable_thinking = None

        proxy_value = (
            proxy
            or os.getenv("QWEN_CHAT_PROXY")
            or os.getenv("HTTPS_PROXY")
            or os.getenv("https_proxy")
            or None
        )

        return cls(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=float(os.getenv("QWEN_CHAT_TEMPERATURE", "0") or 0),
            max_tokens=int(os.getenv("QWEN_CHAT_MAX_TOKENS", "8192") or 8192),
            timeout_s=float(
                os.getenv("QWEN_CHAT_TIMEOUT_SECONDS", "120") or 120
            ),
            max_retries=int(os.getenv("QWEN_CHAT_MAX_RETRIES", "3") or 3),
            enable_thinking=enable_thinking,
            proxy=proxy_value,
        )


class QwenTargetClient:
    """OpenAI-compatible target client with local-image conversion."""

    def __init__(self, config: QwenTargetConfig | None = None) -> None:
        self.config = config or QwenTargetConfig.from_env()
        self._http_client = (
            httpx.Client(
                proxy=self.config.proxy,
                timeout=self.config.timeout_s,
            )
            if self.config.proxy
            else None
        )
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
            http_client=self._http_client,
        )

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        proxy: str | None = None,
    ) -> "QwenTargetClient":
        return cls(QwenTargetConfig.from_env(model=model, proxy=proxy))

    def generate(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        stage: str = "target",
    ) -> LLMResponse:
        del stage  # target client currently has one stage

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._convert_messages(messages),
            "max_tokens": int(max_tokens or self.config.max_tokens),
            "temperature": (
                self.config.temperature
                if temperature is None
                else float(temperature)
            ),
        }

        extra_body = dict(self.config.extra_body)
        if self.config.enable_thinking is not None:
            template_kwargs = dict(
                extra_body.get("chat_template_kwargs") or {}
            )
            template_kwargs["enable_thinking"] = self.config.enable_thinking
            extra_body["chat_template_kwargs"] = template_kwargs
        if response_format is None:
            response_format = self.config.response_format
        if response_format is not None:
            payload["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(max(1, self.config.max_retries)):
            try:
                # ``extra_body`` is an OpenAI SDK keyword that merges its
                # entries into the request body.  Sending it as a JSON field
                # would leave ``chat_template_kwargs`` nested and ignored.
                response = self.client.chat.completions.create(
                    **payload,
                    extra_body=extra_body or None,
                )
                text = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                return LLMResponse(
                    text=str(text),
                    model=self.config.model,
                    usage=TokenUsage(
                        calls=1,
                        prompt_tokens=int(
                            getattr(usage, "prompt_tokens", 0) or 0
                        ),
                        completion_tokens=int(
                            getattr(usage, "completion_tokens", 0) or 0
                        ),
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt + 1 >= self.config.max_retries:
                    break
                time.sleep(min(2 ** attempt, 10))

        raise RuntimeError(
            f"Qwen target call failed after "
            f"{self.config.max_retries} attempts: {last_error!r}"
        ) from last_error

    @staticmethod
    def _convert_messages(messages: Sequence[Message]) -> list[Message]:
        converted: list[Message] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                converted.append(dict(message))
                continue

            parts: list[dict[str, Any]] = []
            for part in content:
                part_type = part.get("type")
                if part_type == "image_path":
                    path = str(part.get("path", ""))
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(path)},
                    })
                else:
                    parts.append(dict(part))

            converted.append({**message, "content": parts})
        return converted

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
        if self._http_client is not None:
            self._http_client.close()


def _image_to_data_url(path: str) -> str:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {path}")

    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


__all__ = [
    "QwenTargetClient",
    "QwenTargetConfig",
]
