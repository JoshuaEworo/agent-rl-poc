# Ling 3.0 Tiny + AppWorld agentic RL PoC

This repository is the smallest useful before/after experiment for one question: does native LoRA + agentic GSPO improve `inclusionAI/Ling-3.0-tiny` on held-out, stateful AppWorld tasks?

It uses AppWorld's real persistent Python environment and database-state evaluators. The model receives one `execute_python` tool, discovers AppWorld APIs through `apis.api_docs`, authenticates and acts over multiple turns, observes execution results and errors, and must call `apis.supervisor.complete_task()` when finished. No ground-truth solution, required-API list, or reference trajectory is shown to the policy.

## Why GSPO

As audited on 2026-09-15, AReno supports both GRPO and GSPO, but its default and documented agentic LoRA path uses GSPO. AReno also natively supports Bailing-MoE V3 LoRA when `no_kda_lora=true`; Ling Tiny's current model config satisfies that constraint. The PoC therefore uses GSPO and AReno's own agent trajectory and PEFT-compatible adapter save/reload paths, with no custom RL algorithm.

Audited upstream revisions:

- AReno: `48d07c54051c41bf36218f99bce1c3697e9ba63c`
- AppWorld: `42b5bcf3cd334fee33f0c37c02070a9f5807add5`
- Ling model: `e3a47d5b986e7141b6efd62597d598ebb392060d`

## Run in Google Colab

1. Push this directory to GitHub, open `ling_appworld_rl.ipynb` from GitHub in Colab, and set the `REPO_URL` cell to that repository URL (or set the `LING_RL_REPO_URL` environment variable).
2. Choose **Runtime → Change runtime type → NVIDIA GPU**. An A100 40 GB or larger is recommended. The notebook prints the GPU, VRAM, CUDA, and PyTorch versions and refuses to train below its conservative 30 GiB smoke-test threshold.
3. If Hugging Face requires authentication for the model, add a Colab secret named `HF_TOKEN`. The notebook copies it to the environment without printing it. AppWorld itself needs no external API key.
4. Run all cells top-to-bottom. Upstream installs can take a while because AReno builds CUDA extensions. If Colab asks for a runtime restart after dependency installation, restart once and resume at **Validate imports**; do not rerun the install cells.
5. The smoke flow runs one inspectable real trajectory, a 2-task held-out baseline, one GSPO update from 2 training tasks × 2 rollouts, verifies the saved adapter files, starts a new server with that adapter (the reload check), and evaluates the same held-out task IDs.

No large training run starts by default. For the larger experiment, change `CONFIG_PATH` in the notebook from `configs/smoke.yaml` to `configs/experiment.yaml`. That preset uses 50 training tasks, 4 rollouts per task, 50 updates, and 25 held-out dev tasks. Adjust task counts, rollout count, LoRA rank, turn limit, microbatching, context length, and generation length directly in the YAML.

Artifacts are written beneath `artifacts/smoke/` or `artifacts/experiment/`:

- `split_manifest.json`: exact, seeded, disjoint task IDs
- `*_trajectories.jsonl`: full model/tool/observation/evaluation traces
- `*_metrics.json`: success, failure, efficiency, reward, and failure categories
- `adapter/step_*/`: PEFT-compatible LoRA adapter
- `comparison.json`: baseline versus trained summary

The final notebook cell can copy the artifact directory to Google Drive if Drive is mounted.

## Local CPU checks

The training engine requires Linux, CUDA, and a supported NVIDIA GPU. CPU-only development can validate all model-independent behavior:

```bash
python -m pip install -e '.[dev]'
pytest
python scripts/train.py --config configs/smoke.yaml --dry-run
```

After `appworld install && appworld download data`, dataset preparation and a real environment construction can also run on CPU. Baseline/trained evaluation expects an AReno OpenAI-compatible server:

```bash
areno serve --model-path inclusionAI/Ling-3.0-tiny --world-size 1 --tp-size 1 --port 8000
python scripts/train.py --config configs/smoke.yaml --dry-run  # prepares split manifest
python scripts/evaluate.py --config configs/smoke.yaml --base-url http://127.0.0.1:8000 --label baseline
```

## Interpreting the result

The primary metric is held-out task success rate from AppWorld's state evaluator. A positive smoke-test delta only proves wiring and provides a weak signal; it is far too small for a scientific conclusion. Use the larger preset and report a paired confidence interval across several seeds before investing. Also inspect average tool calls, reward distribution, and failure categories: improved success accompanied by runaway action counts is not an unqualified win.

AppWorld environments are deliberately serialized inside each AReno batch because their in-process database bindings are global. This sacrifices rollout throughput but prevents sampled rollouts from contaminating one another—the right tradeoff for a cheap validity-first PoC.

## Current upstream references

- [AReno repository and agentic/LoRA usage](https://github.com/inclusionAI/AReno)
- [AppWorld repository and environment/evaluator API](https://github.com/StonyBrookNLP/appworld)
- [Ling 3.0 Tiny model](https://huggingface.co/inclusionAI/Ling-3.0-tiny)

