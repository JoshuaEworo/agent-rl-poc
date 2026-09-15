from __future__ import annotations

from pathlib import Path
from typing import Any


EXECUTE_PYTHON_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Execute Python in AppWorld's persistent shell. The `apis` object is available. "
            "Use `apis.api_docs` to discover APIs, print values to observe them, and call "
            "`apis.supervisor.complete_task(...)` only after the requested state is achieved."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to execute."}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


class AppWorldEnv:
    """Thin lifecycle wrapper around the real AppWorld environment."""

    def __init__(self, experiment_name: str, output_root: str | Path | None = None) -> None:
        self.experiment_name = experiment_name
        self.output_root = str(output_root) if output_root else None
        self.world: Any | None = None
        self.task_id: str | None = None

    def reset(self, task_id: str) -> dict[str, Any]:
        from appworld import AppWorld

        self.close()
        if self.output_root:
            from appworld.common.path_store import path_store

            path_store.update_root(self.output_root)
        kwargs: dict[str, Any] = {"task_id": task_id, "experiment_name": self.experiment_name}
        self.world = AppWorld(**kwargs)
        self.task_id = task_id
        return {
            "task_id": task_id,
            "instruction": self.world.task.instruction,
            "datetime": self.world.task.datetime.isoformat(),
            "allowed_apps": list(self.world.task.allowed_apps),
            "db_version": self.world.task.db_version,
        }

    def get_tools(self) -> list[dict[str, Any]]:
        return [EXECUTE_PYTHON_TOOL]

    def execute(self, tool_call: dict[str, Any]) -> str:
        if self.world is None:
            raise RuntimeError("reset() must be called before execute()")
        name, arguments = parse_tool_call(tool_call)
        if name != "execute_python":
            raise ValueError(f"unknown tool: {name}")
        return str(self.world.execute(arguments["code"]))

    def is_done(self) -> bool:
        return bool(self.world and self.world.task_completed())

    def evaluate(self) -> dict[str, Any]:
        if self.world is None:
            raise RuntimeError("no active world")
        return dict(self.world.evaluate().to_dict(stats_only=False))

    def close(self) -> None:
        if self.world is not None:
            self.world.close()
            self.world = None

    def __enter__(self) -> AppWorldEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def parse_tool_call(call: Any) -> tuple[str, dict[str, Any]]:
    import json

    if hasattr(call, "model_dump"):
        call = call.model_dump()
    if not isinstance(call, dict):
        raise ValueError("tool call must be an object")
    function = call.get("function", call)
    name = function.get("name")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid tool arguments JSON: {exc}") from exc
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError("tool call needs a function name and object arguments")
    if name == "execute_python" and not isinstance(arguments.get("code"), str):
        raise ValueError("execute_python requires a string `code` argument")
    return name, arguments
