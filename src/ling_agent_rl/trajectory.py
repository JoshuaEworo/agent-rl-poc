from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Step:
    turn: int
    model_response: dict[str, Any]
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    observation: str | None = None
    error: str | None = None
    reward: float | None = None


@dataclass(slots=True)
class Trajectory:
    task_id: str
    user_request: str
    environment: dict[str, Any]
    model_config: dict[str, Any]
    steps: list[Step] = field(default_factory=list)
    reward: float = 0.0
    success: bool = False
    terminal_reason: str = "unknown"
    evaluation: dict[str, Any] = field(default_factory=dict)
    failure_category: str | None = None

    @property
    def turns(self) -> int:
        return len({step.turn for step in self.steps})

    @property
    def tool_calls(self) -> int:
        return sum(step.tool_name is not None for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["turns"] = self.turns
        result["tool_calls"] = self.tool_calls
        return result


def append_jsonl(path: str | Path, trajectory: Trajectory) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trajectory.to_dict(), ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
