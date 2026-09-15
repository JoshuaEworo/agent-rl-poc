from __future__ import annotations

from typing import Any


def environment_reward(success: bool, tool_calls: int, step_penalty: float = 0.01) -> float:
    """Binary objective reward with a small, bounded efficiency penalty."""
    return (1.0 if success else 0.0) - min(max(tool_calls - 1, 0) * step_penalty, 0.25)


def reward_fn(record: Any) -> float:
    """AReno hook; result is attached by the real AppWorld rollout."""
    source = record.source_record
    results = source.get("_rollout_results", {})
    sample_index = str(record.metadata.get("sample_index", 0))
    result = results.get(sample_index)
    if result is None:
        return 0.0
    return float(result["reward"])

