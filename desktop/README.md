# Training Studio

A native C# Windows app for local CUDA fine-tuning through WSL.
The app uses .NET 10 WPF and the Windows 11 Fluent theme. The source has an MIT license.

## Install

Download the Windows x64 ZIP from the repository's releases. Extract all files.
Run `install.ps1` with PowerShell. It installs the app in your Windows user profile
and adds Desktop and Start menu shortcuts. No administrator access is required.
You can also run `TrainingStudio.exe` directly from a permanent folder.
The release includes .NET. It is not code-signed, so Windows can show an unknown-publisher prompt.

## Connect your backend

This first release requires an already prepared WSL backend. It does not install CUDA,
Docker, Python packages or model weights for you. Follow the repository's
[WSL setup](https://github.com/M1chlCZ/local-coding-assistant/blob/main/research/wsl.md) and [focused training recipe](https://github.com/M1chlCZ/local-coding-assistant/blob/main/research/focused-training.md).
Use a recent NVIDIA driver and enough free space for the model, datasets and checkpoints.

In **Connection**, choose your WSL distribution, Linux user and absolute project folder.
Set the controller Python path, normally `.cache/rlm-env/bin/python`. Select **Save and connect**.
The Linux checkout must include `desktop_bridge.py` from this release.
Settings stay in `%LOCALAPPDATA%\TrainingStudio\connection.json`.

The tested recipe needs `.cache/qwen35-env`, the pinned Qwen3.5-4B weights, a prepared
`.cache/polyglot-runtime.json`, its Docker image, and the cached coding benchmarks.
Dependencies must be prepared before starting: the worker runs with Hugging Face offline mode.
These prerequisites are checked again by the training and evaluation tools.

## Use it

- **Overview** shows worker activity, GPU memory, elapsed time and measured coding results.
- **Activity & checkpoints** shows the current stage log and complete saved checkpoints.
- **Start / Resume** starts saved work in a windowless background worker.
- **Pause for gaming** requests a checkpoint and releases the GPU. Wait for **paused**.
- **Stop** ends the finite experiment. Use Pause if you want to continue it later.

Closing the app leaves the worker running. Saved Pause and Stop remain in effect after login.
Training Studio adds a task named `TrainingStudio-…` only when a worker needs to start.
Existing workers continue to use their existing controls. One global GPU lock prevents
simultaneous training runs; pause the active experiment before starting another.

## Startup preferences

Open **Startup** and select **Save startup preferences** after making changes.

- **Resume learning automatically after Windows login** is on by default. It adds login
  and recovery triggers. Saved Pause, Stop and completed experiments stay respected.
- **Open Training Studio after Windows login** is off by default. Turn it on to show
  the dashboard when you sign in. This switch alone does not start learning.

Turn both off to open Studio yourself and press **Start / Resume** manually.
Disabling automatic learning removes login, timer and failure-retry starts. Changing
these preferences does not stop work already running; use **Pause for gaming** for that.
Settings stay in `%LOCALAPPDATA%\TrainingStudio\startup.json`.
The controls apply to Studio workers, not legacy PowerShell tasks or unrelated apps.

## GPU power

Open **GPU power**, select the NVIDIA GPU, and choose **Enable power control** once.
Windows asks for administrator permission for this setup. Then use the slider or
5% preset buttons and select **Apply training limit**. The percentage uses the GPU's
default wattage, not current utilization. The driver supplies the minimum; presets
above 100% or below that minimum are unavailable.

Studio shows both the pending choice and the actual limit read back from NVIDIA.
It saves the selected limit for the current connection and experiment on this PC.
Changing the slider alone does not change power. **Restore 100%** also saves 100%
as the training setting.

- **Pause** restores 100% while the checkpoint saves. Wait for **paused** before gaming.
- **Resume** reapplies the saved limit when the worker becomes active, within its
  ten-second status interval.
- Stop, failure and completion restore 100%. This also works with the dashboard closed.

The limit affects the whole GPU, including other apps. A force-killed Windows worker
cannot run its cleanup; use **Restore 100%** if its limit remains after a crash.
Driver resets can clear a limit; the running worker checks it again. GPU power setup
and the saved percentage do not start learning or change your login preferences.
Only one experiment/GPU power setting is managed per Windows user at a time.

The one-time setup creates fixed, manually triggered NVIDIA preset tasks in
`\TrainingStudioPower\`. Their administrator-owned permissions allow the normal
user to read and run them, but not change their elevated commands. Each command
uses the Windows driver tool at its system path. No elevated learning worker or
long-running PowerShell console is needed. Unsupported GPUs show a clear message.
This setup requires a Windows administrator account; it does not support entering
a different administrator's credentials for a standard account.

NVIDIA documents the supported power ranges and privileges in its
[nvidia-smi reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

## Create an experiment

Open **New experiment**. This release supports **Qwen3.5-4B BF16 LoRA** on a 16 GB NVIDIA GPU.
Other architectures are not supported by the wizard yet.

Choose a unique name and a Linux path to a training JSONL file.
Its adjacent `<file>.manifest.json` must describe the training split, SHA-256,
language counts and execution provenance accepted by `qwen35_train.load_verified`.
The fixed recipe requires at least 2,000 examples, including 250 Go and 250 TypeScript examples.
Raw chat exports are not accepted as verified training data.

The existing `focused_data.py` importer produces this format from the pinned public dataset:

```sh
.cache/rlm-env/bin/python focused_data.py --output .cache/studio-data
```

Select **Validate and preview data**, then choose a one-, three- or six-hour active training limit.
Select **Create paused experiment**. The app copies the dataset into the private experiment
folder, binds its hashes, and prepares four stages: baseline audit, training, trained audit,
and comparison. Select **Start / Resume** when ready.

The trainer makes at most one data pass. Benchmark time is separate from training time.
Oversized answers are excluded, not truncated. If the time budget expires before training
completes, the checkpoint stays saved and the experiment has no final quality verdict.

## Read the result

**Not yet measured** means matched tests are not complete. Partial test counts are progress.
**Improved on these tests** means more tasks pass without losing earlier passes.
**Mixed results** exposes regressions even when the total rises. **Regressed on these tests**
means fewer tasks pass overall. No adapter is promoted automatically.

The 790 function tests cover Python, Go, TypeScript, Rust and Dart. These public tests
may occur in model pretraining. They do not establish general coding or agent quality.
Publisher execution evidence is not the same as locally rerunning every training example.
Use the language table and gained/lost task IDs in the exported report when reviewing a model.

## Privacy and maintenance

There is no telemetry, cloud upload or network listener in the desktop app.
Connection profiles and worker logs stay in your Windows user profile. WSL datasets,
checkpoints and reports stay in the ignored `.cache` folder. Exported reports can contain
private paths: review them before sharing. The setup-guide button opens GitHub in your browser.

Each release installs in its own version folder. The installer updates shortcuts and future
scheduled launches without interrupting an older running worker. Close the old dashboard
and open the new shortcut after upgrading. Keep the old version folder until its worker exits.
The installer refuses to overwrite the same version while it is running.
Do not update Python source files bound to an active experiment.

To uninstall, pause experiments first, remove the `TrainingStudio-…` and `TrainingStudioUI` scheduled tasks you created,
then delete the app folder and shortcuts. If GPU power control was enabled, restore 100%
and remove the fixed tasks in `\TrainingStudioPower\` with administrator permission. Keep the WSL experiment folders to retain results.

## Build and test

```powershell
dotnet publish desktop/TrainingStudio/TrainingStudio.csproj -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true -o dist
```

The WSL JSON bridge uses explicit actions and argument lists. It does not accept shell commands.
Run bridge tests on Linux or macOS:

```sh
python3 -m unittest test_desktop_bridge test_focused_report test_focused_experiment
```

CI builds the Windows executable and tests the bridge. Real GPU throughput, checkpoint recovery
and WSL integration still need hardware tests. The app's `--smoke <png-path>` option renders
its live connected dashboard for a local UI smoke check; it does not start or stop training.

On a prepared Windows profile with an installed worker, `--check-startup <output>` checks
all four startup combinations and restores the original preferences. It briefly changes
scheduled triggers without starting or stopping learning.

The native `--self-check` covers GPU range validation and pause/stop/failure decisions.
`--check-gpu-power <running-session> <output>` briefly checks supported 85–100% presets
on real hardware, then restores the original saved setting (100% if unset).
