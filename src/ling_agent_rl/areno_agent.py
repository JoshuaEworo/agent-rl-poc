"""Agent entry point loaded by `areno train --agent-fn`."""

from __future__ import annotations

import os
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ling_agent_rl.agent import run_episode  # noqa: E402
from ling_agent_rl.config import RolloutConfig  # noqa: E402
from ling_agent_rl.trajectory import append_jsonl  # noqa: E402


def _rollout_config() -> RolloutConfig:
    return RolloutConfig(
        max_turns=int(os.environ.get("LING_RL_MAX_TURNS", "8")),
        max_tool_calls=int(os.environ.get("LING_RL_MAX_TOOL_CALLS", "12")),
        timeout_seconds=int(os.environ.get("LING_RL_TIMEOUT_SECONDS", "300")),
        max_generated_tokens=int(os.environ.get("LING_RL_MAX_GENERATED_TOKENS", "768")),
        repeated_action_limit=int(os.environ.get("LING_RL_REPEAT_LIMIT", "2")),
        step_penalty=float(os.environ.get("LING_RL_STEP_PENALTY", "0.01")),
    )


async def run_agent(ctx, batch):
    from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, max_retries=0)
    output = Path(os.environ.get("LING_RL_TRAJECTORY_PATH", "artifacts/train_trajectories.jsonl"))
    turns = []
    invalid_items = []
    try:
        # AppWorld maintains process-global DB bindings. Serial episodes guarantee that
        # every GSPO sample starts from an independent task state.
        for item in batch.iter_samples():
            task_id = str(item.record["task_id"])
            try:
                trajectory, raw_turns = await run_episode(
                    client=client,
                    task_id=task_id,
                    model="policy",
                    rollout_config=_rollout_config(),
                    experiment_name=f"ling_rl_train_{task_id}_{item.sample_index}_{uuid.uuid4().hex[:8]}",
                    item=item,
                )
                trajectory.model_config.update(json.loads(os.environ.get("LING_RL_MODEL_CONFIG", "{}")))
                for messages, response in raw_turns:
                    turns.append(AgentTrajectoryTurn(item=item, messages=messages, response=response))
                item.record.setdefault("_rollout_results", {})[str(item.sample_index)] = {
                    "reward": trajectory.reward,
                    "success": trajectory.success,
                }
                append_jsonl(output, trajectory)
            except Exception:
                invalid_items.append(item)
        return AgentTrajectory(turns=turns, invalid_items=invalid_items)
    finally:
        await client.close()
