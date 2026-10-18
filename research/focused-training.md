# Focused Go and TypeScript experiment

This finite experiment trains a text-only Qwen3.5-4B adapter on real repository repairs.
It preserves the previous student and its checkpoints.
No coding improvement is established until all comparisons finish.

## Data and training

The source is [SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2), revision `10483de0f50fe5da545942705a76c6150171af7f`.
The importer keeps repair patches from permissively licensed repositories with successful publisher test records.
Those records do not mean that this project reran every training example.

The dataset contains 2,511 training examples, 200 development examples, and 200 reserved final examples.
Repository families stay in separate splits. Duplicate patches and direct benchmark overlap are excluded.
These checks cannot detect every copied repository or prior model exposure.

At a 4,096-token limit, 2,456 complete training examples fit: 1,453 Go and 1,003 TypeScript.
The other 55 examples are excluded. The trainer does not truncate answers.
Only answer tokens contribute to the training loss.

The model is [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
The isolated environment follows the [Unsloth BF16 LoRA recipe](https://unsloth.ai/docs/models/qwen3.5/fine-tune).
The fixed settings use rank 8, a learning rate of 0.00005, and eight accumulation steps.
One data pass requires 307 optimizer steps. The limit is six active training hours.
The trainer saves a checkpoint every ten steps and when Pause reaches the next optimizer step.

## Comparisons

The worker compares the previous student, the unchanged Qwen3.5 model, and the trained Qwen3.5 adapter.
Each model receives the same prompts, output limits, and greedy sampling settings.

- All 790 HumanEval/MultiPL-E function tasks cover Python, Go, TypeScript, Rust, and Dart.
- A small reserved Go/TypeScript repair pilot runs real repository tests in isolated containers.
- Each eligible repair must fail with the original code and pass with its reference patch.

The report lists new passes and lost passes for each language.
The worker does not automatically replace the accepted model.
The unused final repair split stays reserved.
The previous student uses NF4; Qwen3.5 uses BF16. That comparison includes both model and precision differences.

## Windows controls

After preparation, open the focused control panel:

```powershell
.\learning.cmd -Focused
```

Pause saves progress and releases the GPU. Resume continues from saved work.
Pause and Stop remain saved after Windows login.
A running experiment resumes after login. A five-minute trigger recovers an unexpectedly closed worker.
Completed experiments stay complete. Stop ends this finite experiment.

Training and benchmark clocks are separate. Benchmarks can extend the total time beyond six hours.
The panel shows the current stage, checked tasks, optimizer steps, and actual worker health.
Runtime settings, examples, reports, logs, and weights stay under the ignored `.cache/` directory.

## Local tools

`focused_data.py` prepares the pinned dataset and its provenance records.
`focused_replay.py` checks repository test environments before model evaluation.
`qwen35_train.py` checks complete examples and trains the adapter.
`focused_experiment.py` runs an immutable sequence from a local settings file.
`focused_evaluation.py` saves each completed evaluation batch for recovery.
`focused_report.py` compares matched results and records regressions.

A new machine needs the pinned model, isolated training environment, Docker images, and a reviewed local settings file.
The Windows controls do not download or prepare these dependencies automatically.

## Legacy verified-code recipe

This section records the earlier complete-code recipe for existing sessions.
New 0.1.4 sessions use [execution learning](execution-learning.md), including its
lower initial learning rate, language cap and separate REPL gate.

The earlier continuous recipe keeps up to 128 verified answers per language. The exporter
interleaves languages without reducing every language to the smallest pool.
At least 100 complete examples, including eight in each language, must fit before
training starts. Imported replay stays in the training split and is checked again
with the source tests. Validation and benchmark answers never enter replay.

New blocks use one pass over the data, capped at 128 optimizer steps. The starting
learning rate is 0.00002. If the latest complete full benchmark lost previous
passes, the next block halves that trial's rate, down to 0.000005. A successful
trial keeps its rate. Smaller existing rates are never raised after a regression.
Partial results and small validation results do not change the rate. Each block
records the decision and its source report in `recipe.json`.

This is an experimental update limit, not a promise of better code. Reducing the
rate too far can also prevent learning; the floor prevents endless reductions.
The trainer still saves every five steps and when Pause reaches the next optimizer
step. Resume keeps the started block's exact recipe and restores the optimizer,
scheduler, and random state. Recipe changes apply to unstarted blocks only.

Matched validation runs after every training block. A complete 790-task comparison
runs for a promising candidate, or every 20 actual training blocks for a progress
report. Collection-only rounds do not count. An audit already in progress finishes
from its saved tasks. Promotion still requires a validation gain and no lost passes
on the full comparison. These reused tests do not establish general coding quality.

The settings follow the bounded-epoch and data-quality guidance in the
[Unsloth LoRA guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).
Increasing repeated exposure to a tiny dataset is not a substitute for more useful examples.

Replay verification is cached by the complete answer, test cases, language,
checker source, and immutable compiler image. A change to any of these reruns
the checks. Successful cached checks avoid compiling the same replay on every block.

If the current trainer reports a failure but hangs while closing, the worker
releases that process and uses the existing checkpoint retry path. It checks the
process ID and start time so an old error cannot stop a new run. Pause and Stop
remain in control during recovery.
