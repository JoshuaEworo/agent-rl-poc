from pathlib import Path

import pytest


def test_real_appworld_reset_execute_persist_and_evaluate() -> None:
    pytest.importorskip("appworld")
    if not Path("data/tasks").is_dir():
        pytest.skip("run `appworld download data` for the real CPU integration test")

    from appworld import load_task_ids

    from ling_agent_rl.appworld_env import AppWorldEnv

    task_id = load_task_ids("train")[0]
    with AppWorldEnv("ling_rl_pytest_probe") as env:
        metadata = env.reset(task_id)
        assert metadata["instruction"]
        assert env.execute(
            {"name": "execute_python", "arguments": {"code": "integration_value = 40 + 2\nprint(integration_value)"}}
        ).strip() == "42"
        assert env.execute(
            {"name": "execute_python", "arguments": {"code": "print(integration_value + 1)"}}
        ).strip() == "43"
        evaluation = env.evaluate()
        assert isinstance(evaluation["success"], bool)
        assert evaluation["num_tests"] > 0

