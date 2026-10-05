# Local Coding Assistant

A Windows launcher for English coding chat and agent use on a 16 GB NVIDIA GPU.
It runs llama.cpp on the PC through CUDA, with CPU offloading for larger models.
The system prompt requests English. The weights remain multilingual.

**Research prototype.** The default is Qwen3.8-27B with pinned Unsloth UD-Q4_K_M weights.
Setup downloads the model and runtime from their upstream repositories.
Our pruned Qwen3.6 derivative remains available through the [experiment recipe](research/recipe.md).
Its weights are not published yet.

## Run on Windows

Requirements: Windows x64, Python 3.10+, a recent NVIDIA driver, approximately 32 GB RAM, and 20 GB free disk.

Before setup, stop the model server.
Run these commands:

```powershell
py -3 launcher.py setup
py -3 launcher.py serve --open
```

Alternatively, double-click `start.cmd`.
Setup verifies download hashes. Ctrl+C stops the server.
The default uses an 8K context and eight CPU FFN layers.

If memory runs out, decrease `--context`.
For dense models, increase `--cpu-ffn`.
For MoE models, increase `--cpu-moe`.

If the pruned derivative exists on your PC, run the faster preset:

```powershell
py -3 launcher.py --manifest research/manifest-reap50.json serve --open
```

## Connect an agent

On the PC, open chat at `http://127.0.0.1:8080`.
For Mac access, use the [SSH connection instructions](research/wsl.md#connect-from-the-mac).

| API setting | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8080/v1` |
| Model | `local-coding-assistant` |
| API key, if the client requires one | `local` |

The external agent harness supplies tools and file access.
The API listens on loopback.

## Recursive coding experiment

The [RLM experiment](research/rlm.md) adds a bounded Python REPL and optional model subcalls.
A finite search compares instructions and recursion depth on separate development and holdout tasks.
The [adapter recipe](research/adapter-training.md) runs local QLoRA from successful training traces.
The [confirmation comparison](research/confirmation.md) checked both retained 4B adapters on ten reserved public repairs.
Both passed 9/10, against 1/10 for the unchanged base. The later training updates did not increase this confirmation score.
The [WSL recipe](research/wsl.md) runs the complete experiment on the PC through CUDA.
After WSL setup, `start-rlm.cmd` starts its model server.
The [learning session](research/learning-session.md) collects checked repairs and trains local adapters.
Double-click `learning.cmd` for Windows Start, Pause, Resume, and Stop controls.
Checkpoints retain the optimizer and current step.
[Continuous mode](research/continuous-learning.md) prepares fresh checked public tasks and runs until Pause or Stop.
The [multilingual curriculum](research/polyglot.md) rotates Python, Go, TypeScript, Rust, and Dart with mixed replay and separate language scores.
The latest full audit scored 473/790 for the base model and 433/790 for the RLM adapter.
The revised [balanced code trial](research/polyglot.md#balanced-code-trial) starts each candidate from the base model and trains equal language samples.
It trains verified source answers instead of REPL actions. No quality improvement is established for this new recipe yet.
It recovers after Windows login when previously running. Pause releases the GPU for gaming and remains saved across logins.
The earlier tuned session uses 30 repair tasks, visible tests, and short training candidates.
Candidates receive 15 development checks before acceptance. Three rounds without improvement stop the session.
The [training evidence](research/rlm-training-evidence.md) compares published RLM methods, available weights, and history-derived data.
The [public-data pilot](research/public-data.md) prepares checked Python repairs from NVIDIA OpenCodeInstruct.
Its initial 4B candidate passed 7/10 development tasks, against 5/10 for the previous adapter, and preserved its older passes.
The [Strata pilot](research/strata.md) measures a larger MoE teacher on the same GPU.
The [private history tool](research/private-history.md) exports local Codex messages and requests task ideas from the PC's model.
Exports, reviews, and private adapters stay in the ignored `private-data/` directory.
History-derived ideas need executable tests before training.

## Measured results

The [benchmark guide](research/benchmarking.md) separates training checks, reserved repair tests, and direct coding benchmarks.
Training uses a separate Qwen3-4B student.
On all 164 HumanEval tasks, its unchanged NF4 base passed 118 and the accepted adapter passed 121.
The adapter gained 14 tasks and lost 11, so the higher total does not establish an improvement without regressions.
Continuous mode schedules fresh repair checks and a full HumanEval audit after each experiment, with a six-active-hour limit.
Unchanged accepted weights reuse verified results. The Windows controls show the schedule and last complete comparison.

RTX 5070 Ti, Ryzen 9 9900X, 32 GB RAM, 8K context:

| Model | Coding subset | Practical pilot | Median decode speed |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B, eight CPU FFN layers | 32/32 | 5/5 | 25 tokens/s |
| Our pruned Qwen3.6, no CPU expert offload | 29/32 | 5/5 | 183 tokens/s |
| MiMo V2.6 Distill 9B, no thinking | 20/32 | 4/5 | 117 tokens/s |

These small samples do not establish broad agent reliability.
Different quantization methods prevent an isolated comparison of model architectures.
The [comparison notes](research/candidates.md) record recent releases, memory use, and failures.
On 15 authored REPL repairs, the previous 4B adapter passed 8/15 against its base at 0/15.
The [tuning notes](research/learning-session.md#expanded-baseline-result) separate those baseline results from the revised training run.
The [model card](MODEL_CARD.md) describes our derivative and its limits.

## Reproduce

```powershell
py -3 test_launcher.py
py -3 test_evaluation.py
```

See the [recipe](research/recipe.md) for compression and evaluation commands.
Docker isolates generated code during evaluation.

Before publishing, run `python3 check_public_data.py --history`.
The check detects personal home paths, private history records, and SSH tunnel hosts outside the documented example.
To run it before each push, use `git config core.hooksPath .githooks`.
Review credentials and screenshots separately.

Original code uses the [MIT license](LICENSE).
Upstream model licenses remain separate.
See [NOTICE.md](NOTICE.md) and [ATTRIBUTION.md](ATTRIBUTION.md) for sources and redistribution terms.
