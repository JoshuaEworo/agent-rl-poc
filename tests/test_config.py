from pathlib import Path

from ling_agent_rl.config import load_config


def test_load_smoke_config() -> None:
    config = load_config(Path("configs/smoke.yaml"))
    assert config.model == "inclusionAI/Ling-3.0-tiny"
    assert config.train.algorithm == "gspo"
    assert config.train.rollouts_per_task >= 2
    assert config.train_split != config.eval_split

