# Recursive coding experiment

This experiment uses the [official RLM library](https://github.com/alexzhang13/rlm), pinned to `d04208afbad29ca675ab13478c40ee8bebc84bfe`.
The model inspects repository files through a persistent Python REPL.
It can request focused model analysis or start a nested REPL.
Recursion changes inference. It does not update model weights.

The current setup runs the model through CUDA on the Windows PC.
The Mac runs the experiment controller and Docker sandbox through an SSH relay.
A complete PC setup also requires Docker. WSL installation can require a Windows restart.
The Windows launcher works without WSL.

## Prepare

Requirements: Python 3.11+, Git, Docker, and the local model server.
The container uses the same pinned Python image as the existing evaluator.

Run these commands from the repository folder:

```bash
python -m venv .cache/rlm-env
.cache/rlm-env/bin/python -m pip install -r requirements-rlm.txt
docker pull python@sha256:51dafde81dbdb6ebde285137a295cf18a47ca95234fe388a343719cb97305b3d
.cache/rlm-env/bin/python test_recursive_agent.py
```

On Windows, use `.cache/rlm-env/Scripts/python.exe` instead of `.cache/rlm-env/bin/python`.
The commands use the local server at port 8080.
For a relay, change `--base` to its loopback address.

## Compare and retain settings

```bash
.cache/rlm-env/bin/python autoresearch.py --base http://127.0.0.1:8080 --trials 2 --calls 16 --seconds 240 --holdout --output reports/rlm-round1
```

The first comparison uses a REPL without model subcalls.
Each RLM trial receives the same call, output-token, and time ceilings.
The search changes only a general instruction and recursion depth.
It retains settings by successful repairs, then generated tokens, then elapsed time.
Proposal generation uses one extra model request outside task ceilings.
The final holdout results never enter proposal feedback.

Reports save after each task. Existing output folders cannot receive another run.
The fixtures contain 20 authored repositories: ten training, five development, and five holdout tasks.
Each repository contains two files. These tasks do not establish performance on large codebases.
The public holdout tests provide separation within a run, not a permanently secret benchmark.

## First measurements

Qwen3.8-27B UD-Q4_K_M used the PC's RTX 5070 Ti, an 8K context, and eight CPU FFN layers.

| Setting | Development repairs | Model subcalls |
| --- | ---: | ---: |
| REPL baseline | 3/5 | Disabled |
| Initial RLM setting | 2/5 | 0 |

The model did not use recursion in this round. These results show no improvement.
A second instruction named development tasks. The final proposal rules reject that instruction.
Its partial holdout run is excluded from evaluation evidence.
All submitted patches received another grading run after the sandbox review.
The [reports](../reports/rlm-qwen38-round1/summary.json) preserve the measured results and limits.
Docker checks also exercise ordinary subcalls and a nested child with controlled model responses.
Actual recursive model performance still needs a separate measurement.

## Propose a repository repair

```bash
.cache/rlm-env/bin/python recursive_agent.py --repo /path/to/repository --prompt "Repair the parser" --output reports/proposed-repair.json
```

The snapshot includes Python and Markdown files, with a 2 MiB limit.
The runner outputs complete replacement files in JSON. It does not apply them.
Inspect the proposed patch before application.

## Prepare learning data

```bash
.cache/rlm-env/bin/python recursive_agent.py --split train --output reports/rlm-train.json
python training_data.py reports/rlm-train.json --output .cache/training/round1.jsonl
```

The exporter keeps root traces from successful training tasks.
It excludes development tasks, holdout tasks, subcalls, and duplicate traces.
Successful repairs can contain unsuccessful intermediate reasoning. Inspect the traces before training.
The [adapter recipe](adapter-training.md) prepares a separate, bounded QLoRA run on a small dense model.
No adapter receives automatic promotion or upload.

## Execution limits

Generated Python runs in disposable containers without network access or host mounts.
Containers use a read-only root filesystem, an unprivileged user, and memory, CPU, process, and output limits.
All model calls share one task budget, including nested calls.
The host accepts only bounded model requests and result messages from the sandbox.
The grader performs functional checks. It is not a secure verifier against hostile Python code.

The custom sandbox adapter replaces the upstream environment factory for each run.
The runner also corrects the pinned library's final-answer request role.
Keep the upstream pin when reproducing results.

Sources: [RLM paper](https://arxiv.org/html/2512.24601v3), [RLM training notes](https://github.com/alexzhang13/rlm/tree/main/training), [Karpathy autoresearch](https://github.com/karpathy/autoresearch).
