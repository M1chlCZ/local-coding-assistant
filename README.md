# Local Coding Assistant

A Windows launcher for local coding chat and agent use on a 16 GB NVIDIA GPU.
The model runs on the PC through CUDA and llama.cpp.
The system prompt requests English. The model weights remain multilingual.

**Research prototype.** The default model is a pruned Qwen3.6-35B-A3B derivative.
The repository contains source code, model hashes, and experiment reports.
Derivative weights are not published yet.
For a fresh installation, build them with the [recipe](research/recipe.md) or use the downloadable baseline.

## Run on Windows

Requirements: Windows x64, Python 3.10+, a recent NVIDIA driver, approximately 32 GB RAM, and 15 GB free disk.
The research recipe requires more disk space.

If the derivative exists on your PC, run these commands:

```powershell
py -3 launcher.py setup
py -3 launcher.py serve --open
```

Alternatively, double-click `start.cmd`.
Before setup, stop the model server.
Setup verifies download hashes. Ctrl+C stops the server.

For a fresh PC, run the intact baseline:

```powershell
py -3 launcher.py --manifest research/manifest-unsloth.json setup
py -3 launcher.py --manifest research/manifest-unsloth.json serve --open
```

The baseline needs more disk space and CPU expert offloading.
The default derivative uses an 8K context and no CPU expert offloading.
If memory runs out, decrease `--context` or increase `--cpu-moe`.

## Connect an agent

Chat: `http://127.0.0.1:8080`.

| API setting | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8080/v1` |
| Model | `local-coding-assistant` |
| API key, if the client requires one | `local` |

The external agent harness supplies tools and file access.
The API listens on loopback.

## Measured results

The default retains 128 of 256 routed experts per layer and uses Q4_K_M quantization.
On an RTX 5070 Ti, its 11.37 GB file produced these results:

- 29/32 tasks in a fixed HumanEval subset.
- 5/5 tasks in a small coding and repository repair pilot.
- 183 tokens/s median decode speed.
- Successful synthetic retrieval with a 31,738-token prompt.

The final sampler uses a repetition penalty of 1.1 over 256 tokens.
Tuning used a known failure. These small samples do not establish broad agent reliability.
The [model card](MODEL_CARD.md) records failures, comparison settings, memory use, and limits.

## Reproduce

```powershell
py -3 test_launcher.py
py -3 test_evaluation.py
```

See the [recipe](research/recipe.md) for model compression and evaluation commands.
Docker isolates generated code during evaluation.

Original code uses the [MIT license](LICENSE).
Qwen derivatives use Apache-2.0 with the notices in [NOTICE.md](NOTICE.md) and [ATTRIBUTION.md](ATTRIBUTION.md).
The calibration sources require a separate license review before a public model release.
