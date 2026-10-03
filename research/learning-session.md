# Learning session on Windows

The session runs on the PC through CUDA in the dedicated `LocalCodingAssistant` WSL environment.
An on-demand Windows task owns the worker process. An SSH disconnect does not stop the worker.
The Mac supplies controls and source updates.
For automatic task replenishment and login recovery, use [continuous mode](continuous-learning.md).

Complete the [WSL setup](wsl.md) and [training environment setup](adapter-training.md#prepare-in-wsl) first.

## Start and control

Close the chat model server before you start the session.
Double-click `learning.cmd` to open the Windows panel.
Select **Start** for a session with a 12-hour active limit.
Select **Pause** to save progress and release the GPU.
Select **Resume** to continue the same session.
Select **Stop** to finish the session permanently.
Closing the panel keeps the worker active.

For live PowerShell progress, use:

```powershell
.\learning.cmd -Action watch -Session .cache/learning/research
```

The console shows task counts, training steps, quality checks, model scores, and elapsed time.
Closing this console keeps the worker active. The panel also prints progress in its PowerShell console.
During work, the Windows worker prevents automatic idle sleep.
Pause restores normal idle sleep. The display can turn off.

Alternatively, run these commands from the Windows project folder:

```powershell
.\learning.cmd -Action start -Hours 12
.\learning.cmd -Action status
.\learning.cmd -Action pause
.\learning.cmd -Action resume
.\learning.cmd -Action stop
```

Pause finishes the current bounded repair task or saves the next training step.
The panel shows `paused` after the GPU process exits.
A repair task has a 120-second model budget, plus container setup and cleanup.
Training checkpoints include the adapter, optimizer, scheduler, random state, and step count.
Resume checks the data, settings, and checkpoint hashes before it restores training.

The task uses your current Windows login, with no password or administrator setting.
The launcher gives that account control of its learning task, including tasks created through an elevated SSH connection.
It has no time trigger and does not start at login.
Windows logout or restart stops the worker.
After login, select **Resume** to restore the last complete checkpoint and saved task progress.
An interrupted checkpoint stays excluded until all required files pass the hash checks.
If no complete checkpoint exists, the worker preserves the incomplete output and restarts that training round.
A changed or corrupt complete checkpoint stops the session.
Source changes also stop resume. Restore the original source or select a new session folder.

The time limit counts active work. Paused time and time without a worker do not count.
The worker saves progress at the limit and stops.
The time limit is a ceiling. A session can finish earlier when its fresh training tasks run out.
An empty final batch saves an empty collection report and finishes without another model load or training update.
Resume also recovers this condition from an interrupted export.
The saved status records `completion_reason: curriculum_exhausted` and keeps the accepted checkpoint and elapsed time.
Start and Resume reject a completed or stopped session.
For another session, select a new folder:

```powershell
.\learning.cmd -Action start -Hours 12 -Session .cache/learning/next-session
```

## What the session learns

The first round measures the larger Qwen3.8-27B teacher on all 15 development repairs before student training.
Each round then collects training repairs through the recursive harness.
The curriculum contains 30 distinct training repairs, including 20 new repair families.
The [task file](repair_tasks.json) contains the new training and development repositories.
Visible tests accompany each training repository.
The authored grading tests reject the broken source and accept the reference repair before collection.
Reference repairs check the fixtures. They never supply teacher answers or training targets.

Only teacher repairs that pass the grading tests enter training.
The collector grades each proposed repair in an isolated container.
The exporter checks registry hashes and trusts the recorded result in the local collection report.
It preserves the teacher's actual REPL turns and their recorded execution feedback.
It excludes failed repairs, child calls, and repeated examples.
The previous exporter substituted four scripted turns for each repair. That setting is no longer used by the session.
These examples teach the recorded repair process. They do not establish recursive delegation.
Replay uses up to four recent rounds. Each round fixes its dataset before training.
If a round adds no new examples, the worker stops before another update.
Changing task labels alone does not count as new examples.
Development and holdout tasks never enter the training dataset.

The student uses pinned Qwen3-4B weights, NF4, rank-8 QLoRA, and complete examples of at most 4,096 tokens.
Each candidate has a ceiling of one dataset pass.
Fresh adapters use at most 80 steps at `5e-5`.
Updates from an accepted adapter use at most 20 steps at `5e-6`.
The trainer saves a full checkpoint every five steps and on pause or completion.
It retains three periodic checkpoints per round.
The evaluator checks complete checkpoints from the earliest retained step to the latest.
It keeps the first checkpoint that passes the full improvement rule. The final training step does not receive preference.
A new round uses the last accepted adapter, with a new optimizer.
Resume within a round restores the existing optimizer.
The teacher, trainer, and evaluator use the GPU in separate phases.

By default, every candidate receives the same 15 development repairs as its unchanged base.
The set contains the original five tasks and ten new repositories.
For extra checks, put a development registry in the new session's `development-tasks.json` before its first start.
The session checks and records its hash. Resume rejects changes to that file.
The registry accepts only development tasks with unique IDs. It never supplies training examples.
Acceptance requires more successful repairs than the base and the previous accepted candidate, with no regression on individual tasks.
The worker stops after three consecutive rounds without development improvement.
The repeated development set can overfit. It does not prove general coding quality.

For an explicit research session, add `-Research` to the Windows start command or `--research` to the Linux worker.
Research continues after three rejected updates and retains the active time limit.
It mixes the original verified training round with each fresh round and tries 5, 10, and 20 steps in turn.
Prepare fresh training tasks before each round. Changing old task labels does not supply new problems.
Research can continue from a higher-scoring checkpoint with at most one lost task.
This experimental checkpoint stays separate from the accepted adapter. Promotion still requires improvement without lost passing tasks.
The unchanged base receives one evaluation per session. Later rounds reuse that result.
See [the replay research experiment](replay-research.md).
For optional early rejection and measured runtime limits, see [research speed](performance.md).
This experiment leaves the original holdout untouched. The chat model receives no automatic replacement.
The worker also stops after 64 rounds or three rounds without a successful new repair.
It stops at 20 GiB of session files or less than 10 GiB of free disk space.
No generated datasets or weights upload automatically.

The previous stopped session stays in `.cache/learning/current`.
Its checkpoints require the original source files for resume.
The new Windows panel controls `.cache/learning/tuned` by default.
After an exporter update, use a new session folder. An existing session keeps its original source binding.

## Compare a coding model

Stop all learning and chat processes first.
Run the pinned student, its previous adapter, and a local MiMo GGUF through the same development harness:

```bash
.cache/rlm-env/bin/python research/tuning_compare.py --adapter .cache/learning/current/round-002/adapter --gguf .cache/models/MiMo-V2.6-Distill-Qwen-9B.Q4_K_M.gguf --output .cache/tuning-comparison
```

Use the actual local GGUF path.
The command requires a new output folder and cached student weights.
It owns the GPU lock and stops its model servers on exit.
Each task receives eight model calls, 4,096 output tokens, and 120 seconds.
The result measures this harness. Different model sizes and quantization methods prevent an isolated architecture comparison.

## Files

The Linux session folder is `.cache/learning/tuned` within the project folder.
Open it from Windows at `\\wsl.localhost\LocalCodingAssistant\home\coder\local-coding-assistant\.cache\learning\tuned`.
Each `round-NNN` folder contains teacher traces, a fixed dataset, checkpoints, training logs, and development reports.
`status.json` records the current phase, active time, and accepted candidates.
Windows worker logs stay in `.cache\learning-windows` in the Windows project folder.

Chat uses the same GPU. During training and evaluation, the chat server is unavailable.
After pause or completion, run `start-rlm.cmd` to restore chat.
Close that chat server before you resume learning.

This setup learns supervised REPL repairs. It does not guarantee better recursion or autonomous delegation.
The first adapter trial scored 0/5 development repairs. Longer training alone does not establish improvement.

Sources: [Transformers checkpoints](https://huggingface.co/docs/transformers/main_classes/trainer),
[Trainer callbacks](https://huggingface.co/docs/transformers/main_classes/callback),
[Windows sleep control](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate).

## Control test

For Windows startup checks, run `powershell.exe -NoProfile -File .\test_learning_start.ps1`.
For native failure reporting, run `powershell.exe -NoProfile -File .\test_learning_worker.ps1` after WSL setup.
The latter check creates a temporary task and invalid session. It starts no model and deletes its own task and session.
The worker retains its native process handle so Windows PowerShell reports the actual exit code.
A worker that completes during startup returns its completed status instead of a false launch error.
See the [PowerShell exit-code issue](https://github.com/PowerShell/PowerShell/issues/5421).

The [PC control test](../reports/learning-session-smoke/summary.json) paused at step 20 and resumed after a worker restart.
That earlier run completed all 80 steps and released the GPU while paused.
All ten derived examples in that earlier run fit its token limit.
The base and adapter both scored 0/5 development repairs.
The adapter completed each attempt in two calls, but no repair passed.
The session rejected that candidate. These results establish recovery, not a coding improvement.

## Check checkpoint recovery

To check task access, run `powershell.exe -NoProfile -File .\test_learning_permissions.ps1` from a normal Windows terminal.
This check requires an existing learning task. It updates the same task settings without starting the worker.

For the CPU checks, run `python3 test_training_session.py`.
For the CUDA check, stop all model processes first.
Use an existing verified dataset and its task registry:

```bash
.cache/train-env/bin/python test_training_session.py .cache/learning/tuned/round-001/training.jsonl .cache/learning/tuned/round-001/tasks.json
```

The CUDA check requires the pinned environment and cached Qwen3-4B weights.
It pauses at step one, resumes to step two, and restores a completed checkpoint without another step.
The test uses a temporary output folder and removes its own files afterward.

## Expanded baseline result

The [baseline report](../reports/tuning-2026-10-01/baseline-summary.json) compares 15 development repairs before the revised training run.
The unchanged 4B model passed 0/15. Its previous adapter passed 8/15.
That adapter passed seven new tasks and one original task.
These tasks are authored mini-repositories. This result does not establish general coding reliability.

MiMo 9B passed 0/15 through this REPL interface.
It returned tool-call tags instead of executable REPL blocks.
This result measures interface incompatibility. It does not rank its standalone coding ability.
The [larger Qwen model](../reports/tuning-2026-10-01/teacher-summary.json) passed 11/15 through the compatible interface.
It used eight CPU FFN layers and took 720.38 seconds, against 187.57 seconds for the small adapter.
One larger-model task exceeded its time limit. Neither model made recursive child calls.

All 120 checked trajectory examples fit the token limit. The longest checked example used 1,225 tokens.
These fixture checks used authored reference patches only for execution and token checks.
The training exporter still requires successful teacher repairs.
CUDA checkpoint checks paused at step one and resumed to step two.
Recovery from the completed checkpoint took no additional optimizer step.

The tuned run then stopped after three rejected rounds. Every new candidate scored 0/15.
The previous 8/15 adapter remains selected. Training completion did not establish improvement.
Two later trials used actual teacher traces at learning rates of `5e-5` and `5e-6`.
Both failed the first repair that the previous adapter passed. The early regression check rejected them.
The [public-data pilot](public-data.md) prepares a separate, checked coding curriculum.
