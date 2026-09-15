from __future__ import annotations

import argparse
import json
from pathlib import Path

from ling_agent_rl.evaluate import compare


parser = argparse.ArgumentParser()
parser.add_argument("artifact_dir")
args = parser.parse_args()
root = Path(args.artifact_dir)
result = compare(
    json.loads((root / "baseline_metrics.json").read_text()),
    json.loads((root / "trained_metrics.json").read_text()),
)
(root / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))

