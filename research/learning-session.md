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

Each round collects repairs from the CUDA Qwen3.8-27B teacher through the recursive harness.
The tasks contain ten training-only module variants and four parameterized repair families.
The module variants reuse existing bugs. They do not create new repair semantics.
The authored tests must reject the broken source and accept the reference repair before collection.
Reference repairs check the fixtures. They never supply teacher answers or training targets.

Only teacher repairs that pass the functional checks enter training.
The exporter checks each repair again and derives two short supervised REPL turns: inspect, then repair.
These derived examples replace failed intermediate attempts and copied repository responses.
Replay uses up to four recent rounds. Each round fixes its dataset before training.
Development and holdout tasks never enter the training dataset.

The student uses pinned Qwen3-4B weights, NF4, rank-8 QLoRA, 1,536-token examples, and 80 optimizer steps per round.
The trainer saves a full checkpoint every ten steps and on pause or completion.
It retains three periodic checkpoints per round.
A new round uses the last accepted adapter, when one exists, with a new optimizer.
Resume within a round restores the existing optimizer.
The teacher, trainer, and evaluator use the GPU in separate phases.

Every candidate receives the same five development repairs as its unchanged base.
Acceptance requires more successful repairs than the base and the previous accepted candidate, with no per-task regression.
The small repeated development set can overfit. It does not prove general coding quality.
The holdout stays unused. The chat model receives no automatic replacement.
The worker stops after 64 rounds or three rounds without a successful new repair.
It also stops at 20 GiB of session files or below 10 GiB of free disk space.
No datasets or weights upload automatically.

## Files

The Linux session folder is `.cache/learning/current`.
Open it from Windows at `\\wsl.localhost\LocalCodingAssistant\home\coder\local-coding-assistant\.cache\learning\current`.
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
It completed all 80 steps and released the GPU while paused.
All ten derived examples fit the token limit.
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
.cache/train-env/bin/python test_training_session.py .cache/learning/current/round-001/training.jsonl .cache/learning/current/round-001/tasks.json
```

The CUDA check requires the pinned environment and cached Qwen3-4B weights.
It pauses at step one, resumes to step two, and restores a completed checkpoint without another step.
The test uses a temporary output folder and removes its own files afterward.
