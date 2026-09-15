from __future__ import annotations

import asyncio
import json
import uuid
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from openai import AsyncOpenAI

from .agent import run_episode
from .config import ExperimentConfig
from .trajectory import append_jsonl, read_jsonl


async def evaluate_endpoint(
    config: ExperimentConfig,
    task_ids: list[str],
    *,
    base_url: str,
    api_key: str = "unused",
    label: str = "baseline",
) -> dict[str, Any]:
    client = AsyncOpenAI(base_url=base_url.rstrip("/") + "/v1", api_key=api_key, max_retries=0)
    path = Path(config.artifact_dir) / f"{label}_trajectories.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    try:
        for task_id in task_ids:
            trajectory, _ = await run_episode(
                client=client,
                task_id=task_id,
                model="policy",
                rollout_config=config.rollout,
                experiment_name=f"ling_rl_{label}_{task_id}_{uuid.uuid4().hex[:8]}",
            )
            trajectory.model_config.update({"label": label, "adapter": "none" if label == "baseline" else "reloaded_lora"})
            append_jsonl(path, trajectory)
    finally:
        await client.close()
    metrics = aggregate(read_jsonl(path))
    metrics["trajectory_path"] = str(path)
    metrics_path = Path(config.artifact_dir) / f"{label}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "tasks": 0, "success_rate": 0.0, "failure_rate": 0.0,
            "average_turns": 0.0, "average_tool_calls": 0.0,
            "reward_distribution": [], "failure_categories": {},
        }
    successes = sum(bool(row.get("success")) for row in rows)
    return {
        "tasks": len(rows),
        "success_rate": successes / len(rows),
        "failure_rate": 1.0 - successes / len(rows),
        "average_turns": mean(float(row.get("turns", 0)) for row in rows),
        "average_tool_calls": mean(float(row.get("tool_calls", 0)) for row in rows),
        "reward_distribution": [float(row.get("reward", 0.0)) for row in rows],
        "mean_reward": mean(float(row.get("reward", 0.0)) for row in rows),
        "failure_categories": dict(Counter(row.get("failure_category") for row in rows if not row.get("success"))),
    }


def compare(baseline: dict[str, Any], trained: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "trained": trained,
        "success_rate_delta": trained["success_rate"] - baseline["success_rate"],
        "mean_reward_delta": trained.get("mean_reward", 0.0) - baseline.get("mean_reward", 0.0),
    }


def run_evaluation(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(evaluate_endpoint(*args, **kwargs))
