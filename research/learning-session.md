# Learning session on Windows

The session runs on the PC through CUDA in the dedicated `LocalCodingAssistant` WSL environment.
An on-demand Windows task owns the worker process. An SSH disconnect does not stop the worker.
The Mac supplies controls and source updates.

Complete the [WSL setup](wsl.md) and [training environment setup](adapter-training.md#prepare-in-wsl) first.

## Start and control

Close the chat model server before you start the session.
Double-click `learning.cmd` to open the Windows panel.
Select **Start** for a session with a 12-hour active limit.
Select **Pause** to save progress and release the GPU.
Select **Resume** to continue the same session.
Select **Stop** to finish the session permanently.
Closing the panel keeps the worker active.
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
The exporter checks each repair again in an isolated container.
It runs four supervised REPL turns: inspect, reproduce the failure, apply the repair and retest, then submit the patch.
Test output comes from execution. The exporter rejects repairs that fail the visible retest.
These examples teach test use and repair. They do not demonstrate recursive delegation.
Replay uses up to four recent rounds. Each round fixes its dataset before training.
Development and holdout tasks never enter the training dataset.

The student uses pinned Qwen3-4B weights, NF4, rank-8 QLoRA, and complete examples of at most 4,096 tokens.
Each candidate trains for approximately one dataset pass, with an 80-step ceiling and a learning rate of `5e-5`.
The trainer saves a full checkpoint every five steps and on pause or completion.
It retains three periodic checkpoints per round.
A new round uses the last accepted adapter, with a new optimizer.
Resume within a round restores the existing optimizer.
The teacher, trainer, and evaluator use the GPU in separate phases.

Every candidate receives the same 15 development repairs as its unchanged base.
The set contains the original five tasks and ten new repositories.
Acceptance requires more successful repairs than the base and the previous accepted candidate, with no regression on individual tasks.
The worker stops after three consecutive rounds without development improvement.
The repeated development set can overfit. It does not prove general coding quality.
This experiment leaves the original holdout untouched. The chat model receives no automatic replacement.
The worker also stops after 64 rounds or three rounds without a successful new repair.
It stops at 20 GiB of session files or less than 10 GiB of free disk space.
No generated datasets or weights upload automatically.

The previous stopped session stays in `.cache/learning/current`.
Its checkpoints require the original source files for resume.
The new Windows panel controls `.cache/learning/tuned` by default.

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
The teacher comparison in the tuned session supplies a larger model with a compatible interface.

All 120 checked trajectory examples fit the token limit. The longest checked example used 1,225 tokens.
These fixture checks used authored reference patches only for execution and token checks.
The training exporter still requires successful teacher repairs.
CUDA checkpoint checks paused at step one and resumed to step two.
Recovery from the completed checkpoint took no additional optimizer step.
