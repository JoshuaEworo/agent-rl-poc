from pathlib import Path

from ling_agent_rl.config import load_config
from ling_agent_rl.train import areno_command


def test_command_requires_real_lora_and_agentic_gspo() -> None:
    config = load_config("configs/smoke.yaml")
    command = areno_command(config, Path("tasks.jsonl"))
    joined = " ".join(command)
    assert "--algo gspo" in joined
    assert "--agent-fn" in command
    assert "--lora-rank" in command
    assert "--save-interval 1" in joined
    assert "--model-hub hf" in joined

