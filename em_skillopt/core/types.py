"""Core data types shared by the EM target harness and the SkillOpt bridge.

This module intentionally depends only on the Python standard library.  It is
the contract between:

- the EM episode runner (target side)
- the four domain adapters
- the SkillOpt adapter / reflection / metrics code

Keeping it dependency-free makes it importable from a lightweight optimizer
environment and from every EmbodiedBench conda environment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence, TypeAlias

# A single low-level action.  High-level environments use ints (action ids) or
# strings; EB-Manipulation uses a 7-dimensional list of ints.
ActionValue: TypeAlias = int | str | list[int]

# OpenAI-style chat message.  We keep this as a plain dict so the same object
# can be passed to the EM RemoteModel and to the SkillOpt Qwen backend.
Message: TypeAlias = dict[str, Any]

# multi inheritance
class Split(str, Enum):
    """Dataset split names used by the runner and dataloaders"""
    # enum values are defined as:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @classmethod
    def coerce(cls, value: "Split | str") -> "Split":
        """force inversion
        args:
            @cls: class type
            @value: a Split object or str
        """
        normalized = str(value).strip().lower()
        # typical alias mapping to standard names
        alias = {
            "training": "train",
            "selection": "val",
            "valid": "val",
            "valid_seen": "val",
            "dev": "val",
            "valid_unseen": "test",
            "evaluation": "test",
        }
        return cls(alias.get(normalized, normalized))

def json_safe(value: Any) -> Any:
    """
    Recursively convert any Python object into basic types that can be serialized by json.dumps, while guaranteeing no exceptions are raised.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # recursively process 'Mapping' type
    # key in json must be 'str' while python's dict's key can be int,tuple etc
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    # json do not have tuple/set
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    tolist = getattr(value, "to_list", None)
    if callable(tolist):
        return json_safe(tolist(value))
    return str(value)

@dataclass
class TokenUsage:
    """Token accounting for one or more LLM calls."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TokenUsage":
        data = data or {}
        return cls(
            calls=int(data.get("calls", 0) or 0), # key may exist, but value = None / "" / False
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
        )

@dataclass
class LLMResponse:
    """Normalize LLM response returned by every LLM client"""
    text: str
    model: str = ""
    # `default_factory` accepts a no-argument callable, which is called once each time an instance is created, returning a new object.
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_s: float = 0.0
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "latency_s": self.latency_s,
            "raw": json_safe(dict(self.raw)),
        }

@dataclass
class EpisodeSpec:
    """One episode to execute.

    ``index`` is the position in the original EmbodiedBench dataset, not the
    position in the current split.  The domain adapter needs it to move the
    simulator to the right episode.
    """

    episode_id: str
    env_name: str
    index: int
    instruction: str
    capability: str = "base"
    split: Split = Split.TRAIN
    max_steps: int = 30
    seed: int = 42
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "env_name": self.env_name,
            "index": self.index,
            "instruction": self.instruction,
            "capability": self.capability,
            "split": self.split.value,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "metadata": json_safe(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeSpec":
        return cls(
            episode_id=str(data["episode_id"]),
            env_name=str(data["env_name"]),
            index=int(data["index"]),
            instruction=str(data.get("instruction", "")),
            capability=str(data.get("capability", "base")),
            split=Split.coerce(data.get("split", "train")),
            max_steps=int(data.get("max_steps", 30)),
            seed=int(data.get("seed", 42)),
            metadata=dict(data.get("metadata") or {}),
        )
