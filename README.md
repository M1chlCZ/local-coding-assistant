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

Chat: `http://127.0.0.1:8080`.

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
The [adapter recipe](research/adapter-training.md) prepares local QLoRA from successful training traces.
Adapter training and its GPU memory use remain unverified.
The [WSL recipe](research/wsl.md) runs the complete experiment on the PC through CUDA.
After WSL setup, `start-rlm.cmd` starts its model server.

## Measured results

RTX 5070 Ti, Ryzen 9 9900X, 32 GB RAM, 8K context:

| Model | Coding subset | Practical pilot | Median decode speed |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B, eight CPU FFN layers | 32/32 | 5/5 | 25 tokens/s |
| Our pruned Qwen3.6, no CPU expert offload | 29/32 | 5/5 | 183 tokens/s |
| MiMo V2.6 Distill 9B, no thinking | 20/32 | 4/5 | 117 tokens/s |

These small samples do not establish broad agent reliability.
Different quantization methods prevent an isolated comparison of model architectures.
The [comparison notes](research/candidates.md) record recent releases, memory use, and failures.
The [model card](MODEL_CARD.md) describes our derivative and its limits.

## Reproduce

```powershell
py -3 test_launcher.py
py -3 test_evaluation.py
```

See the [recipe](research/recipe.md) for compression and evaluation commands.
Docker isolates generated code during evaluation.

Original code uses the [MIT license](LICENSE).
Upstream model licenses remain separate.
See [NOTICE.md](NOTICE.md) and [ATTRIBUTION.md](ATTRIBUTION.md) for sources and redistribution terms.
