# Public coding data pilot

This pilot uses [NVIDIA OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct).
The dataset uses CC BY 4.0. Credit: NVIDIA, OpenCodeInstruct.
We change selected examples into Python repair tasks. This does not change the upstream license.

The importer reads the first 1,000 rows of one pinned shard.
It accepts short standalone Python functions with at least two tests and complete passing test metadata.
It runs each reference solution in an isolated container, then checks that the incomplete solution fails.
Only `solution.py` is editable. The visible tests stay fixed.
An exact content hash removes repeated tasks and sets the training or development split.
This rule separates exact duplicates. It does not detect every similar problem.

The local check kept **208 tasks: 163 training and 45 development**.
Metadata or syntax excluded 790 rows. Isolated execution excluded two more.
Passing the supplied tests does not prove general correctness.
Overlap with model pretraining data is unknown.
The original project holdout remains unused. Private chat history supplies no examples.

## Reproduce in WSL

Complete the [WSL setup](wsl.md) first. Start Docker and stop the model servers.
Use a separate data environment; the training environment does not need PyArrow.

```bash
python3 -m venv .cache/data-env
.cache/data-env/bin/pip install pyarrow==23.0.1
mkdir -p .cache/public-data
curl --fail --location --output .cache/public-data/source.parquet \
  https://huggingface.co/datasets/nvidia/OpenCodeInstruct/resolve/8f3ba5bafe4d6e8db46082cf7ae6741bc370604d/data/train-00000-of-00050.parquet
.cache/data-env/bin/python public_data.py --parquet .cache/public-data/source.parquet --output .cache/public-data/verified
```

The importer checks SHA-256 before it reads the shard:
`342757e0c6b706c8f68cf0a867ead5c417d0273726453dbcfc934f5c3c6ca891`.
Use a new output folder for each run.
If disk space is limited, remove `source.parquet` after a successful conversion.
Keep the snapshot, task registry, and manifest for reproducibility.
All these files stay under ignored `.cache/`.

The task registry contains reference answers for fixture checks.
The collector supplies only the task request, incomplete files, and editable paths to the teacher.
Training uses only teacher repairs that pass independent container checks.
It exports actual root turns through `training_data.py`; development tasks never enter training.
The exporter checks the registry and report hashes. It trusts the grading result in each local report.
With the teacher running, collect the training split:

```bash
.cache/rlm-env/bin/python recursive_agent.py --tasks .cache/public-data/verified/tasks.json --split train --depth 2 --calls 8 --output-tokens 4096 --seconds 120 --output .cache/public-data/teacher-train.json
python3 training_data.py .cache/public-data/teacher-train.json --tasks .cache/public-data/verified/tasks.json --output .cache/public-data/training.jsonl
```

This command collects all 163 training tasks. The measured short pilot used the first 32 training tasks and ten development tasks.
Stop the teacher before student training.
Use [the adapter recipe](adapter-training.md) for checkpointed CUDA training.
Pass the public task registry through `--tasks` when you train or check the exported dataset.
Compare each candidate with the base and the previous adapter before accepting it.

## CUDA student trial

The short trial selected 32 training tasks and ten separate development tasks.
Qwen3.8-27B passed 30/32 training tasks and 10/10 development tasks.
Its successful repairs supplied 123 actual root turns. All examples fit the 4,096-token limit.
No recursive child calls occurred.

The candidate started from the previous Qwen3-4B adapter, with NF4 and rank-8 QLoRA.
It trained for 20 optimizer steps at `5e-6`, with a one-pass ceiling.
The trainer saved complete checkpoints with optimizer state.

| Student | Public development repairs | Authored development repairs |
| --- | ---: | ---: |
| Unchanged 4B base | 0/10 | 0/15 from the previous baseline |
| Previous adapter | 5/10 | 8/15 from the previous baseline |
| Public-data candidate | 7/10 | 9/15 |

The candidate preserved every previous passing task in both development sets.
It also passed a second complete authored evaluation after a model reload.
The [aggregate report](../reports/public-data-2026-10-02/summary.json) records hashes, settings, memory use, and selection results.
The candidate stays local. It does not replace the 27B chat server.
The original holdout and the other 35 public development tasks remain unused.
These small, repeated development sets do not establish general coding reliability.
