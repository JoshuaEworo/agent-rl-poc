import json

import pytest

from ling_agent_rl.appworld_env import AppWorldEnv, EXECUTE_PYTHON_TOOL, parse_tool_call


def test_environment_wrapper_initializes_without_loading_heavy_runtime() -> None:
    env = AppWorldEnv("unit-test")
    assert env.experiment_name == "unit-test"
    assert env.world is None
    assert env.get_tools() == [EXECUTE_PYTHON_TOOL]


def test_tool_schema_is_openai_function() -> None:
    function = EXECUTE_PYTHON_TOOL["function"]
    assert function["name"] == "execute_python"
    assert function["parameters"]["required"] == ["code"]
    json.dumps(EXECUTE_PYTHON_TOOL)


def test_parse_openai_tool_call() -> None:
    name, arguments = parse_tool_call(
        {"function": {"name": "execute_python", "arguments": '{"code":"print(1)"}'}}
    )
    assert name == "execute_python"
    assert arguments == {"code": "print(1)"}


def test_parse_rejects_bad_action() -> None:
    with pytest.raises(ValueError, match="string"):
        parse_tool_call({"name": "execute_python", "arguments": {"code": 3}})
