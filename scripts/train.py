from __future__ import annotations

import argparse
import shlex

from ling_agent_rl.config import load_config
from ling_agent_rl.train import print_runtime, train


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/smoke.yaml")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
config = load_config(args.config)
print_runtime()
command = train(config, dry_run=args.dry_run)
print("AReno command:\n" + shlex.join(command))

