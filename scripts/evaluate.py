from __future__ import annotations

import argparse
import json
from pathlib import Path

from ling_agent_rl.config import load_config
from ling_agent_rl.evaluate import run_evaluation


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/smoke.yaml")
parser.add_argument("--base-url", default="http://127.0.0.1:8000")
parser.add_argument("--label", choices=["baseline", "trained"], default="baseline")
args = parser.parse_args()
config = load_config(args.config)
manifest = json.loads((Path(config.artifact_dir) / "split_manifest.json").read_text(encoding="utf-8"))
metrics = run_evaluation(config, manifest["eval_task_ids"], base_url=args.base_url, label=args.label)
print(json.dumps(metrics, indent=2))

