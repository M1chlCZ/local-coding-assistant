# Model research: 30 September 2026

Kimi Code CLI researched alternatives with the `kimi-code/k3` model.
We independently verified the official model pages and pinned GGUF file metadata.
Published benchmark scores do not predict results on this PC.

## Recent candidates

We sorted official Hugging Face author listings by `createdAt`, newest first.
We screened coding-capable releases separately from quantizations and unrelated model types.
These timestamps show repository creation, not a guaranteed public release date.
The scoped metadata snapshot is [recent-models.json](recent-models.json).

| Official repository | Created on Hugging Face | Local fit |
| --- | --- | --- |
| [MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) | 21 September 2026 | 5.63 GB Q4_K_M, dense coding and agent SFT |
| [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 10 September 2026 | Too large intact |
| [GLM-5.3 Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 25 August 2026 | Too large intact, approximately 321B parameters |
| [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 24 August 2026 | Too large intact |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 5 August 2026 | 16.46 GB Unsloth UD Q4_K_M, with CPU offload |

MiMo is a recent SFT of Qwen3.5-9B on MiMo-generated data.
It is not a new foundation architecture or a separable set of coding experts.
Its card and quantization metadata state MIT, but the pinned source tree lacks a LICENSE file.
A model redistribution needs the applicable upstream license texts and notices.
This code repository references downloads and does not redistribute those weights.

Qwen3.8-27B is dense.
The launcher supports `--cpu-ffn` for dense feed-forward layers and `--cpu-moe` for MoE expert layers.
The Qwen3.8 comparison manifest starts with eight CPU FFN layers.

## Older comparison candidates

| Candidate | Artifact | File size | License | Evidence |
| --- | --- | ---: | --- | --- |
| GPT-OSS-20B | Official ggml-org MXFP4 | 12.11 GB | Apache-2.0 | [OpenAI card](https://huggingface.co/openai/gpt-oss-20b), [GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) |
| Gemma 4 12B | Official Google QAT Q4_0 | 6.98 GB | Apache-2.0 | [Google card and GGUF](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf), [license](https://ai.google.dev/gemma/apache_2) |
| Devstral Small 2 24B | Unsloth Q4_K_M | 14.33 GB | Base: Apache-2.0. Mirror metadata: other. | [Mistral card](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512), [GGUF](https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF) |

GPT-OSS uses MoE and the Harmony chat format.
Gemma supplies a smaller dense baseline with more memory headroom.
Devstral Q4_K_M leaves little space for the context cache on a 16 GB GPU.
That fit assessment uses file sizes, not measured allocation.
We did not benchmark these older candidates in this round after the user requested recent releases.

GPT-OSS and Gemma use different quantization methods from our Qwen derivative.
Their comparison measures complete deployments, not model architecture alone.
The recent-model manifests record file hashes, revisions, and runtime defaults.

## Larger recent models

[DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) has 763B total parameters and a roughly 510 GB checkpoint.
[Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) has 125B language parameters, plus 51B embedding parameters and 4B MTP parameters.
Active parameter counts describe computation, not the full weight storage requirement.
Ordinary four-bit quantization does not fit these intact models into this PC's GPU and RAM.
Expert pruning cannot remove all shared tensors or guarantee coding quality.
We did not download, prune, or benchmark either large model.


## Results on the test PC

RTX 5070 Ti, Ryzen 9 9900X, 32 GB RAM, Windows 11, llama.cpp b11146.
All coding trials used greedy first attempts and the same fixed 32 tasks.
The practical pilot allowed one feedback repair per coding task and a bounded repository repair.
The sampler used repetition penalty 1.1 over 256 tokens.
Thinking trials retained the same output budgets and prompts.

| Deployment at 8K context | HumanEval subset | Practical pilot | Median decode speed |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B UD-Q4_K_M, eight CPU FFN layers, no thinking | 32/32 | 5/5 | 25.18 tokens/s |
| MiMo V2.6 Distill 9B Q4_K_M, no CPU offload, no thinking | 20/32 | 4/5 | 116.98 tokens/s |
| MiMo V2.6 Distill 9B Q4_K_M, no CPU offload, thinking enabled | 21/32 | 3/5 | 117.17 tokens/s |
| Previous pruned Qwen3.6 default, no CPU expert offload | 29/32 | 5/5 | 183 tokens/s |

MiMo emitted tool calls without available tools on some code-only tasks.
It also missed ordinary edge cases.
Thinking improved one subset result but reduced the practical score.
We retained every failed response.
The embedded template matched the official MiMo template except for a trailing newline.
The model uses the supported dense `qwen35` architecture.

Qwen3.8 passed the supplied tools through the API and repaired the repository fixture.
It is the downloadable launcher default after this round.
Our pruned derivative remains the faster research preset.
These results describe small local trials, not general model rankings or independent training achievements.

At 8K, Qwen3.8 retrieved a value from 7,158 prompt tokens in 5.48 seconds to first text.
At 32K, it retrieved the value from 31,738 tokens in 28.24 seconds with sixteen CPU FFN layers.
That longer-context request decoded at 16.32 tokens/s.
Synthetic retrieval does not measure coding quality across large repositories.

MiMo retrieved the same value at 8K in 1.46 seconds and at 32K in 6.76 seconds.
It needed no CPU offload in either context trial.

Sampled total GPU use reached 6,635 MiB for MiMo at 8K and 15,746 MiB for Qwen3.8 at 8K.
Qwen3.8 reached 15,834 MiB during its 32K request.
Available system RAM briefly fell to 0.54 GB during the Qwen3.8 8K run.
The tasks completed, but this 32 GB system had little RAM headroom at that point.
These one-second samples include other Windows applications and can miss short peaks.
Monitoring started during the Qwen3.8 benchmark and ended before the MiMo 32K request.
The [comparison summary](../reports/recent-comparison.json) records sample counts and report hashes.

## Run the recent candidates

Stop the current server before setup or a model switch.

```powershell
py -3 launcher.py setup
py -3 launcher.py serve --open
```

For 32K context:

```powershell
py -3 launcher.py serve --context 32768 --cpu-ffn 16 --open
```

For the MiMo comparison:

```powershell
py -3 launcher.py --manifest research/manifest-mimo26-9b.json setup
py -3 launcher.py --manifest research/manifest-mimo26-9b.json serve --open
```

For evaluation, use the PC API at `http://127.0.0.1:8080`.
These commands require Docker on the evaluation machine.

```powershell
py -3 benchmark.py --base http://127.0.0.1:8080 --label candidate --output reports/humaneval-candidate.json
py -3 evaluation.py --base http://127.0.0.1:8080 --context 8192 --label candidate --output reports/pilot-candidate.json
```

Add `--thinking` to each evaluation command for a separate thinking trial.
The reports in `reports/` retain the model outputs and runtime properties.
