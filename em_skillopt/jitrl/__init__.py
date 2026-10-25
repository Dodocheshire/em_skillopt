"""JitRL-style test-time adaptation for EM-Benchmark (Method B).

This package implements a logit-free adaptation of JitRL.  The official
implementation corrects token logits from retrieved trajectory advantages.
The remote Qwen endpoint used by EM-Benchmark does not expose logits, so this
version applies the same memory/advantage idea at the level of a parsed action
plan: retrieved action returns are used to reorder the plan and, optionally,
insert a high-value memory action.
"""
from .memory import JitRLMemory, JitRLRecord
from .policy import JitRLConfig, JitRLPlanAdjuster

__all__ = [
    "JitRLConfig",
    "JitRLMemory",
    "JitRLPlanAdjuster",
    "JitRLRecord",
]
