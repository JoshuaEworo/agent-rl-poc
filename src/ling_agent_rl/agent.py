from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .appworld_env import AppWorldEnv, parse_tool_call
from .config import RolloutConfig
from .rewards import environment_reward
from .trajectory import Step, Trajectory


SYSTEM_PROMPT = """You are operating a real AppWorld task through a persistent Python shell.
Work iteratively: inspect API documentation when needed, authenticate, execute API calls, and use
printed results as observations. Errors are observations; correct course and retry. Never guess that
the task is done. When the requested real-world state is achieved, call
apis.supervisor.complete_task() (include answer=... for answer-seeking tasks). Do not ask the user
questions. Keep code short and inspect outputs before consequential calls."""


def assistant_message(response: Any) -> dict[str, Any]:
    message = response.choices[0].message
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return dict(message)


def classify_failure(terminal_reason: str, steps: list[Step], evaluation: dict[str, Any]) -> str | None:
    if evaluation.get("success"):
        return None
    if terminal_reason in {"timeout", "max_turns", "max_tool_calls", "repeated_action"}:
        return terminal_reason
    if any(step.error for step in steps):
        return "tool_or_environment_error"
    if not any(step.tool_name for step in steps):
        return "no_tool_call"
    return "incorrect_final_state"


async def run_episode(
    *,
    client: Any,
    task_id: str,
    model: str,
    rollout_config: RolloutConfig,
    experiment_name: str,
    item: Any | None = None,
) -> tuple[Trajectory, list[tuple[list[dict[str, Any]], Any]]]:
    """Run one genuine multi-turn AppWorld episode against an OpenAI endpoint."""
    areno_turns: list[tuple[list[dict[str, Any]], Any]] = []
    with AppWorldEnv(experiment_name) as env:
        metadata = env.reset(task_id)

        # Episode timeout should cover agent interaction,
        # not AppWorld initialization.
        started = time.monotonic()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": metadata["instruction"]},
        ]
        trajectory = Trajectory(
            task_id=task_id,
            user_request=metadata["instruction"],
            environment=metadata,
            model_config={"model": model, "adapter": "endpoint-selected"},
        )
        recent_actions: list[str] = []
        terminal_reason = "max_turns"

        for turn_number in range(1, rollout_config.max_turns + 1):
            if time.monotonic() - started >= rollout_config.timeout_seconds:
                terminal_reason = "timeout"
                break
            request_messages = [dict(message) for message in messages]
            try:
                remaining = max(1.0, rollout_config.timeout_seconds - (time.monotonic() - started))
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=request_messages,
                        tools=env.get_tools(),
                        tool_choice="auto",
                        max_tokens=rollout_config.max_generated_tokens,
                        temperature=rollout_config.temperature,
                        stream=False,
                    ),
                    timeout=remaining,
                )
            except TimeoutError:
                trajectory.steps.append(Step(turn=turn_number, model_response={}, error="model request timed out"))
                terminal_reason = "timeout"
                break
            except Exception as exc:
                trajectory.steps.append(Step(turn=turn_number, model_response={}, error=str(exc)))
                terminal_reason = "model_error"
                break

            areno_turns.append((request_messages, response))
            assistant = assistant_message(response)
            messages.append(assistant)
            calls = assistant.get("tool_calls") or []
            if not calls:
                trajectory.steps.append(Step(turn=turn_number, model_response=assistant))
                terminal_reason = "model_terminated"
                break

            for call in calls:
                step = Step(turn=turn_number, model_response=assistant)
                trajectory.steps.append(step)
                if trajectory.tool_calls >= rollout_config.max_tool_calls:
                    step.error = "maximum tool-call limit reached"
                    step.observation = step.error
                    terminal_reason = "max_tool_calls"
                    break
                try:
                    name, arguments = parse_tool_call(call)
                    signature = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)
                    recent_actions.append(signature)
                    if len(recent_actions) >= rollout_config.repeated_action_limit and len(
                        set(recent_actions[-rollout_config.repeated_action_limit :])
                    ) == 1:
                        terminal_reason = "repeated_action"
                        observation = "Stopped: identical action repeated too many times."
                    else:
                        observation = env.execute(call)
                    step.tool_name = name
                    step.tool_arguments = arguments
                except Exception as exc:
                    observation = f"{type(exc).__name__}: {exc}"
                    step.error = observation
                step.observation = observation
                call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id or f"call_turn_{turn_number}",
                        "name": step.tool_name or "execute_python",
                        "content": observation,
                    }
                )
            if terminal_reason in {"repeated_action", "max_tool_calls"}:
                break
            if env.is_done():
                terminal_reason = "appworld_terminal"
                break

        try:
            evaluation = env.evaluate()
        except Exception as exc:
            evaluation = {"success": False, "evaluation_error": str(exc)}
            if terminal_reason == "appworld_terminal":
                terminal_reason = "environment_error"
        trajectory.evaluation = evaluation
        trajectory.success = bool(evaluation.get("success", False))
        trajectory.terminal_reason = terminal_reason
        trajectory.reward = environment_reward(
            trajectory.success, trajectory.tool_calls, rollout_config.step_penalty
        )
        if trajectory.steps:
            trajectory.steps[-1].reward = trajectory.reward
        trajectory.failure_category = classify_failure(terminal_reason, trajectory.steps, evaluation)
        return trajectory, areno_turns
