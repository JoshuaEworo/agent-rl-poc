from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

from .config import ExperimentConfig


def select_task_ids(config: ExperimentConfig) -> tuple[list[str], list[str]]:
    from appworld import load_task_ids

    rng = random.Random(config.seed)
    train_ids = list(load_task_ids(config.train_split))
    eval_ids = list(load_task_ids(config.eval_split))
    rng.shuffle(train_ids)
    rng.shuffle(eval_ids)
    selected_train = train_ids[: config.train.train_tasks]
    selected_eval = eval_ids[: config.train.eval_tasks]
    overlap = set(selected_train) & set(selected_eval)
    if overlap:
        raise RuntimeError(f"train/evaluation leakage detected: {sorted(overlap)}")
    return selected_train, selected_eval


def prepare_dataset(config: ExperimentConfig) -> tuple[Path, list[str], list[str]]:
    from appworld import AppWorld

    train_ids, eval_ids = select_task_ids(config)
    artifact_dir = Path(config.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = artifact_dir / "train_tasks.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for task_id in train_ids:
            with AppWorld(task_id=task_id, experiment_name="ling_rl_dataset_metadata") as world:
                record = {"task_id": task_id, "instruction": world.task.instruction}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (artifact_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "seed": config.seed,
                "train_split": config.train_split,
                "eval_split": config.eval_split,
                "train_task_ids": train_ids,
                "eval_task_ids": eval_ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_path, train_ids, eval_ids


def areno_command(config: ExperimentConfig, dataset_path: Path) -> list[str]:
    package = Path(__file__).resolve().parent
    output = Path(config.artifact_dir).resolve()
    train = config.train
    return [
        "areno", "train",
        "--ckpt", config.model,
        "--model-hub", "hf",
        "--dataset-path", str(dataset_path),
        "--dataset-loader-fn", str(package / "dataset.py"),
        "--reward-fn-path", str(package / "rewards.py"),
        "--agent-fn", str(package / "areno_agent.py"),
        "--algo", train.algorithm,
        "--world-size", "1", "--tp-size", "1",
        "--batch-size", str(train.batch_size),
        "--mini-bs", str(train.micro_batch_size),
        "--gradient-accumulation-steps", str(train.gradient_accumulation),
        "--n-samples", str(train.rollouts_per_task),
        "--max-running-prompts", "1",
        "--max-prompt-tokens", str(train.max_prompt_tokens),
        "--max-new-tokens", str(config.rollout.max_generated_tokens),
        "--max-context-len", str(train.max_context_length),
        "--lora-rank", str(train.lora_rank),
        "--lora-alpha", str(train.lora_alpha),
        "--lora-target-modules", ",".join(train.lora_targets),
        "--lr", str(train.learning_rate),
        "--max-steps", str(train.max_steps),
        "--save-path", str(output / "adapter"),
        "--save-interval", "1",
        "--activation-checkpointing",
        "--drop-rollout-state",
    ]


def train(config: ExperimentConfig, *, dry_run: bool = False) -> list[str]:
    dataset_path = Path(config.artifact_dir).resolve() / "train_tasks.jsonl"
    if not dry_run:
        dataset_path, _, _ = prepare_dataset(config)
    command = areno_command(config, dataset_path)
    if dry_run:
        return command
    env = os.environ.copy()
    env.update(
        {
            "LING_RL_MAX_TURNS": str(config.rollout.max_turns),
            "LING_RL_MAX_TOOL_CALLS": str(config.rollout.max_tool_calls),
            "LING_RL_TIMEOUT_SECONDS": str(config.rollout.timeout_seconds),
            "LING_RL_MAX_GENERATED_TOKENS": str(config.rollout.max_generated_tokens),
            "LING_RL_REPEAT_LIMIT": str(config.rollout.repeated_action_limit),
            "LING_RL_STEP_PENALTY": str(config.rollout.step_penalty),
            "LING_RL_TRAJECTORY_PATH": str(Path(config.artifact_dir).resolve() / "train_trajectories.jsonl"),
            "LING_RL_MODEL_CONFIG": json.dumps(
                {
                    "base_model": config.model,
                    "algorithm": config.train.algorithm,
                    "adapter": "native_lora",
                    "lora_rank": config.train.lora_rank,
                    "lora_alpha": config.train.lora_alpha,
                    "lora_targets": config.train.lora_targets,
                }
            ),
        }
    )
    subprocess.run(command, check=True, env=env)
    return command


def verify_adapter(path: str | Path) -> Path:
    root = Path(path)
    candidates = sorted(root.glob("step_*")) if root.is_dir() else []
    adapter = candidates[-1] if candidates else root
    required = [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"adapter save/reload check failed; missing: {missing}")
    metadata = json.loads(required[0].read_text(encoding="utf-8"))
    if not metadata.get("r") or not metadata.get("target_modules"):
        raise ValueError("adapter metadata lacks LoRA rank or targets")
    return adapter


def print_runtime() -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc
    info: dict[str, object] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info.update({"gpu": properties.name, "vram_gib": round(properties.total_memory / 2**30, 1)})
    print(json.dumps(info, indent=2))
    return info
