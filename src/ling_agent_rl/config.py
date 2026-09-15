from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RolloutConfig:
    max_turns: int = 8
    max_tool_calls: int = 12
    timeout_seconds: int = 300
    max_generated_tokens: int = 768
    repeated_action_limit: int = 2
    temperature: float = 0.8
    step_penalty: float = 0.01


@dataclass(slots=True)
class TrainConfig:
    algorithm: str = "gspo"
    train_tasks: int = 2
    eval_tasks: int = 2
    rollouts_per_task: int = 2
    max_steps: int = 1
    batch_size: int = 1
    micro_batch_size: int = 1
    gradient_accumulation: int = 1
    lora_rank: int = 4
    lora_alpha: int = 8
    learning_rate: float = 1e-6
    max_prompt_tokens: int = 1536
    max_context_length: int = 8192
    lora_targets: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "f_proj", "g_proj", "o_proj",
            "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "dense",
        ]
    )


@dataclass(slots=True)
class ExperimentConfig:
    model: str = "inclusionAI/Ling-3.0-tiny"
    seed: int = 2026
    train_split: str = "train"
    eval_split: str = "dev"
    artifact_dir: str = "artifacts/smoke"
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rollout = RolloutConfig(**data.pop("rollout", {}))
    train = TrainConfig(**data.pop("train", {}))
    config = ExperimentConfig(rollout=rollout, train=train, **data)
    if config.train.algorithm != "gspo":
        raise ValueError("This PoC intentionally supports AReno's GSPO path only")
    if config.train.rollouts_per_task < 2:
        raise ValueError("GSPO needs at least two rollouts per prompt for relative advantages")
    if config.rollout.max_turns < 1 or config.rollout.max_tool_calls < 1 or config.rollout.repeated_action_limit < 1:
        raise ValueError("rollout limits must be positive")
    return config
