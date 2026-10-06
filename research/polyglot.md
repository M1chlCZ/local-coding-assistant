# Multilingual learning

## Balanced code trial

The `balanced-code-v1` recipe trains complete source answers from the pinned base model.
It does not warm-start from the earlier RLM adapter or train on its Python-only REPL anchor.
Each full round collects four public training examples per language.
A failed example gets one retry with visible feedback. Only answers that pass all registered checks enter training.
The dataset contains equal numbers of answers for Python, Go, TypeScript, Rust, and Dart.
Token filtering keeps this balance and drops complete examples that exceed 4,096 tokens.
Training uses rank-8 CUDA QLoRA, a `1e-5` learning rate, one data pass, and at most 20 optimizer steps.
Only assistant source tokens receive a training loss.

Direct development and confirmation checks support GPU batches of 1, 2, 4, 8, or 16 independent prompts.
The server splits long batches before CUDA allocation. Each GPU batch allows 32,768 padded input and output tokens in total.
Each prompt keeps its 8,192-token limit, including output.
At most two compiler checks run in parallel. Teacher collection and model training keep one GPU owner.
Pause releases the model for gaming. Resume keeps complete saved batches and task order.
Set the controller's `generation_batch_size` while the reviewed worker is idle; use the same size for both models.
A changed size gets a fresh matched baseline in the next experiment. Do not mix serial and batched quality scores.
The full audit also supports `--batch-size 2`, `4`, `8`, or `16`, with a separate source-bound cache.
Shared generation time is counted once per batch when reporting throughput.

Development and fresh confirmation checks use direct source requests with one attempt and no feedback retry.
The old RLM scores are separate measurements. The new baseline uses a zero-delta adapter that matches the base model.
Every accepted candidate must improve its complete development score without losing an earlier passing task.
The RLM remains an optional agent harness. This recipe does not establish RLM agent quality.

Preparing an existing worker requires Pause, an idle worker, and an archived source generation.
Keep unused confirmation tasks reserved before training. Then prepare the separate branch in the training environment:

```bash
.cache/train-env/bin/python code_recipe.py --controller .cache/learning/continuous --previous-source .cache/learning/continuous/source-updates/REVIEWED-UPDATE/source-generation
```

Each new experiment permits six active hours. The original experiment, checkpoints, and global clock remain intact.
Windows Resume starts the prepared branch. Pause and Stop keep their existing behavior.
The full standardized benchmark remains a report-only audit after each experiment.
Do not train on its problems or use its failed tasks as correction targets.

The previous full audit scored 473/790 for the base model and 433/790 for the RLM adapter.
It did not establish a general coding improvement. The revised recipe is an unproven experiment.

## Earlier RLM recipe

The optional curriculum covers Python, Go, TypeScript, Rust, and Dart.
It uses the same local Qwen3-4B student and CUDA QLoRA training.
The larger local model creates repair traces through the RLM's Python REPL.
The repaired source uses the selected programming language.

Each 16-example round starts with eight examples in its main language.
Eight examples cover the other languages. The main language rotates each round.
Training also replays the original verified anchor and the last five rounds.

Each experiment has a six-active-hour limit, followed by confirmation and full audits.
Teacher collection allows 180 seconds per task. Matched development checks keep their 120-second limit.
Multilingual warm updates use a learning rate of `2.5e-6`, half the previous rate.
This is a conservative recipe trial after updates lost earlier passes. It does not establish an improvement.
The 5, 10, and 20-step trials, 4,096-token limit, replay, and acceptance rules stay the same.

## Data and checks

The loop downloads pinned [CodeContests](https://github.com/google-deepmind/code_contests) training shards.
It selects short standard-input/output problems with bounded tests and Python 3 reference solutions.
The reference must pass the source tests in Docker before the problem is kept.
The teacher sees the statement and public examples. Separate source tests grade its answers.
Only successful teacher traces enter training.

A failed training attempt receives one feedback retry.
The retry sees the previous candidate and visible execution errors. Hidden grading output stays private.
Time-limit feedback asks for fewer model calls and less repeated context.
Each attempt allows at most three minutes, capped by the experiment's remaining active time.
A correction enters training only after it passes the registered checks.
The original failed response is retained privately and never becomes a positive training target.
Two failed attempts remain excluded. Infrastructure failures use worker recovery instead of model feedback.

The attempt journal survives Pause, shutdown, and restart.
Resume grades a saved response or continues the pending retry without repeating completed model calls.
The Windows display shows the current attempt and recovered corrections for the round.
Development, confirmation, and standardized benchmark tasks never use this retry path.

All language versions of one problem have the same train or development split.
A problem ledger prevents exact reuse across downloaded windows.
This does not establish independence from pretraining or other public datasets.

The first continuation reserves five development problems per language and four fresh confirmation problems per language.
It also keeps the previous Python development checks.
A complete matched starting evaluation precedes training.
An accepted checkpoint must increase the complete development score and preserve every earlier passing task.
Partial fast-rejection reports never count as complete scores.

## Standardized audits

After each whole experiment, the worker runs fresh matched confirmation checks, then a complete direct coding audit.
The audit contains 164 original Python HumanEval tasks and these pinned MultiPL-E translations:

| Language | Tasks |
| --- | ---: |
| Go | 154 |
| TypeScript | 159 |
| Rust | 156 |
| Dart | 157 |

Each model gets one greedy attempt per task, with the same output limit and compiler image.
The report shows each language, gained and lost tasks, and an equal-weight average of language scores.
These audit tasks never enter training or checkpoint selection.
Identical weights and audit settings reuse a verified complete report.
Gaming Pause interrupts the audit; Resume continues saved task progress.

These function and contest checks do not measure complete applications, framework knowledge, or reliable repository work.
No multilingual quality improvement has been established yet.

## Compiler setup

Use the dedicated Linux training environment. Generated programs run in Docker with no network or host mounts.
The container has read-only system files, a non-root user, and limits on memory, processes, output, and execution time.
Native binaries run only inside its temporary workspace.

```bash
mkdir -p .cache/polyglot-build
docker build -t local-coding-assistant-polyglot -f research/Polyglot.Dockerfile .cache/polyglot-build
.cache/rlm-env/bin/python -c 'import subprocess; from pathlib import Path; from train_adapter import atomic_json; from training_data import sha256; atomic_json(Path(".cache/polyglot-runtime.json"), {"image": subprocess.check_output(["docker","image","inspect","--format","{{.Id}}","local-coding-assistant-polyglot"],text=True).strip(), "dockerfile_sha256": sha256("research/Polyglot.Dockerfile")})'
.cache/data-env/bin/python polyglot_data.py --pool .cache/learning/continuous/polyglot-pool
```

The source files pin SDK images and npm package versions. Debian build packages resolve at image-build time.
The saved image ID is bound to every multilingual session and benchmark; changing it requires a reviewed continuation.
Preparing an existing continuous worker requires Pause, an idle worker handoff, and a verified merged curriculum manifest.
Use `polyglot_session.py --help` for the preparation interface. Keep the native Windows controls for normal operation.
A source update must preserve the old session, source files, accepted checkpoint, and active time budget.

## Attribution and privacy

CodeContests code uses source-specific Apache-2.0 or MIT terms; non-code materials use CC-BY-4.0.
Keep Google DeepMind and original source attribution. Check the original source terms before distributing derived data or weights.
MultiPL-E and HumanEval use MIT terms.
Downloaded problems, model files, raw traces, and session reports stay under the ignored `.cache` directory.
