# Results and verification status

No benchmark result is claimed yet. This checkout does not have the Linux NVIDIA runtime needed to load Ling and execute an AReno update.

## VERIFIED

- Current upstream source was inspected on 2026-09-15 at the revisions listed in `README.md`.
- AReno registers Ling's `bailing_hybrid` architecture through its Bailing-MoE V3 adapter.
- Ling Tiny's current Hugging Face config declares `BailingMoeV3ForCausalLM` and `no_kda_lora: true`, satisfying AReno's native LoRA guard.
- AReno's current CLI exposes agent functions, GSPO, native LoRA targets, adapter-only saves, and adapter reload/serve.
- AppWorld exposes persistent `world.execute(...)`, terminal status, and state-based `world.evaluate()`.
- CPU tests cover configuration, schema conversion, action parsing, serialization, reward lookup, termination classification, aggregation, comparison, and required AReno CLI flags.
- The generated smoke command was inspected and contains `--agent-fn`, `--algo gspo`, `--lora-rank`, attention-only Bailing targets, and one-step adapter saving.
- AppWorld 0.2.0.dev0 was installed from the pinned revision in an isolated Windows CPU environment; its data bundle installed successfully.
- A real train task (`82e2fac_1`) reset through `AppWorldEnv`, two shell calls proved state persistence (`42` then `43`), and the untouched task's real evaluator returned `success=false` with two tests as expected.

## NOT YET VERIFIED

- Actual Ling inference against one AppWorld task.
- One real GSPO optimizer update with nonzero LoRA gradients.
- Adapter save and reload on GPU.
- Baseline and post-training held-out success rates.
- A100 peak VRAM and wall-clock duration.

These are intentionally performed by `ling_appworld_rl.ipynb`; populate this file with the emitted `comparison.json` only after a real run.

## BLOCKED

- GPU verification is blocked in this local Windows/CPU authoring environment. AReno's CUDA backend requires Linux/WSL2, CUDA-enabled PyTorch 2.6+, `nvcc`, and an NVIDIA GPU. The first global AppWorld install also hit a local `py.test.exe` replacement conflict; an isolated virtual environment avoided it.

## Decision rule

The 2-task smoke preset is a wiring check, not evidence of improvement. Proceed to the small experiment only if the smoke run completes, creates a valid adapter, reloads it, and produces objective AppWorld evaluations. Treat the idea as promising only if the larger, multi-seed paired comparison improves held-out success with uncertainty bounds that do not look like task-sampling noise, while action counts and failure modes remain acceptable.
