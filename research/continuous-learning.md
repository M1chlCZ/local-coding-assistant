# Continuous learning on Windows

Continuous mode runs on the PC until you select Pause or Stop.
The Mac chat does not supervise it.
A local supervisor owns fresh data preparation, bounded experiments, retries, and saved control state.

Each experiment retains its time and storage limits.
When it finishes, the supervisor prepares another experiment with fresh checked tasks.
The overall training clock carries forward. Continuous mode has no overall 12-hour cutoff.

## Prepare

Complete the [WSL setup](wsl.md) and [training setup](adapter-training.md) first.
Use an existing research session with an accepted adapter, a verified first-round training anchor, and a completed base evaluation.
Do not change its bound trainer source files.

In the Linux project folder, install the separate data reader:

```bash
python3 -m venv .cache/data-env
.cache/data-env/bin/pip install pyarrow==23.0.1
.cache/rlm-env/bin/python continuous_learning.py init --adopt .cache/learning/research
```

Replace the example session with your prepared session.
Initialization excludes identities from existing public task registries and reserved checks.
The current teacher and student models remain unchanged.

## Windows controls

From the Windows project folder:

```powershell
.\learning.cmd -Continuous
.\learning.cmd -Continuous -Action watch
```

- **Start** starts the local supervisor.
- **Pause** saves progress and unloads learning models for gaming.
- **Resume** continues from saved progress.
- **Stop** ends this controller permanently. It retains saved work.

Wait for **Paused** before starting a game.
A repair can finish its bounded task before pausing. Training saves at the next optimizer step.
Closing the panel or progress console keeps learning active.

The Windows task starts after login without a stored password or elevated execution.
It resumes only when the saved command permits running.
Pause and Stop survive login. Pause also restores normal idle sleep.

Before launching its long-lived worker, Windows probes drive access and Docker readiness in fresh WSL processes.
It waits up to two minutes for boot dependencies. Native WSL startup timeouts also receive retries.
If the worker itself still sees an unavailable drive, it exits so Windows can start a fresh process.
It does not spend repeated training attempts inside a filesystem view without the model drive.
See the [dedicated WSL settings](wsl.md#2-install-docker) to disable Windows PATH translation warnings.

Temporary failures use increasing retry delays, with six consecutive attempts at most.
Integrity failures block immediately. The status explains the required action.
Windows can retry a failed supervisor process three times.
A source change, exhausted approved source, or persistent failure can require attention.

## Data and quality

The source manifest pins all 50 [NVIDIA OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) shards and their file hashes.
This dataset uses CC-BY-4.0. Keep NVIDIA attribution with derived data.
The reader keeps one verified shard and a persistent row cursor.
A local identity ledger excludes used rows and repositories, including reserved evaluation tasks.
Reference repairs must pass isolated tests. The broken fixtures must fail them.
Dataset code runs in Docker with the existing isolation rules.
These checks do not prove semantic novelty or general correctness.

Each fresh experiment uses at most 256 training tasks in batches of 16.
It replays the original verified anchor and reserves 20 separate confirmation tasks before training.
The accepted adapter must improve the complete development result without losing earlier passes.
Partial early-rejection reports never count as a complete score.
An experimental adapter stays separate from the accepted adapter.

After each experiment, the supervisor compares the frozen starting adapter, the final accepted adapter, and the unchanged base on the reserved checks.
Identical adapter weights share one evaluation. Interrupted comparisons resume from completed tasks.
If the final adapter loses a previously passing confirmation task, the next experiment returns to the frozen starting adapter.
Consumed confirmation tasks never enter training.
Repeated development checks can overfit. Continuous updates do not guarantee continuous quality gains.

After the fresh comparison, the supervisor runs the [full HumanEval audit](benchmarking.md#automatic-schedule).
It evaluates the adapter selected for the next experiment, including a required rollback.
Each experiment has a 12-active-hour limit. An earlier experiment end also starts the audit.
Unchanged accepted weights reuse a verified complete result without GPU generation.
Interrupted audits retain finished tasks. Resume and login recovery continue them automatically.
HumanEval reports trends and does not supply training targets or select checkpoints.
The Windows progress display shows the next audit, current task count, and last complete comparison.

## Storage and privacy

State, data, reports, logs, and adapters stay under ignored `.cache/` folders.
Nothing uploads automatically.
The supervisor retains two recent owned experiments and the adapter files they require.
It saves compact historical results before deleting older completed owned experiments.
Adopted sessions and other projects stay intact. Unfinished experiments remain resumable.
Less than 10 GiB free disk blocks further work.
Each experiment retains the 20 GiB storage ceiling.

## Validation

```bash
.cache/rlm-env/bin/python -m unittest test_continuous_learning test_learning_bootstrap test_scheduled_benchmark
```

On Windows:

```powershell
powershell.exe -NoProfile -File .\test_learning_continuous.ps1
powershell.exe -NoProfile -File .\test_learning_bootstrap.ps1
```

These checks cover durable control, retry limits, historical accounting, fresh curriculum handoff, identity exclusions, recovery, retention, and Windows task settings.
They do not simulate a physical reboot or establish model quality.
