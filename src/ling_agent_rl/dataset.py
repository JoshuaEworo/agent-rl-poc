from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict[str, Any]]:
    del default_loader
    records = []
    with Path(dataset_path).open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            records.append(
                {
                    "id": raw["task_id"],
                    "task_id": raw["task_id"],
                    "prompt": raw["instruction"],
                }
            )
    return records

