from types import SimpleNamespace

from ling_agent_rl.agent import classify_failure
from ling_agent_rl.evaluate import aggregate, compare
from ling_agent_rl.rewards import environment_reward, reward_fn
from ling_agent_rl.trajectory import Step, Trajectory, append_jsonl, read_jsonl


def test_trajectory_jsonl_round_trip(tmp_path) -> None:
    trajectory = Trajectory(
        task_id="x_1",
        user_request="do x",
        environment={"seed": 1},
        model_config={"model": "ling"},
        steps=[Step(1, {"content": "x"}, "execute_python", {"code": "print(1)"}, "1")],
        success=True,
        reward=1.0,
        terminal_reason="appworld_terminal",
    )
    path = tmp_path / "trajectories.jsonl"
    append_jsonl(path, trajectory)
    row = read_jsonl(path)[0]
    assert row["turns"] == 1
    assert row["tool_calls"] == 1


def test_reward_and_areno_hook() -> None:
    assert environment_reward(True, 3, 0.01) == 0.98
    record = SimpleNamespace(
        source_record={"_rollout_results": {"2": {"reward": 0.75}}},
        metadata={"sample_index": 2},
    )
    assert reward_fn(record) == 0.75


def test_termination_failure_categories() -> None:
    assert classify_failure("max_turns", [Step(1, {})], {}) == "max_turns"
    assert classify_failure("model_terminated", [Step(1, {})], {}) == "no_tool_call"
    assert classify_failure("appworld_terminal", [], {"success": True}) is None


def test_aggregate_and_compare() -> None:
    baseline = aggregate([
        {"success": False, "turns": 2, "tool_calls": 1, "reward": 0, "failure_category": "no_tool_call"},
        {"success": True, "turns": 4, "tool_calls": 3, "reward": 0.98},
    ])
    trained = aggregate([{"success": True, "turns": 2, "tool_calls": 2, "reward": 0.99}])
    result = compare(baseline, trained)
    assert baseline["success_rate"] == 0.5
    assert baseline["average_turns"] == 3
    assert result["success_rate_delta"] == 0.5

