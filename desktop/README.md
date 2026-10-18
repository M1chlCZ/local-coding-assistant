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

Version 0.1.3 requires a prepared WSL backend. It does not install CUDA,
Docker, Python packages or model weights for you. Follow the repository's
[WSL setup](https://github.com/M1chlCZ/local-coding-assistant/blob/main/research/wsl.md) and [focused training recipe](https://github.com/M1chlCZ/local-coding-assistant/blob/main/research/focused-training.md).
Use a recent NVIDIA driver and enough free space for the model, datasets and checkpoints.

In **Settings → Connection**, choose your WSL distribution, Linux user and absolute project folder.
Set the controller Python path, normally `.cache/rlm-env/bin/python`. Select **Save and connect**.
The Linux checkout must include `desktop_bridge.py` from this release.
Settings stay in `%LOCALAPPDATA%\TrainingStudio\connection.json`.

The tested recipe needs `.cache/qwen35-env`, the pinned Qwen3.5-4B weights, a prepared
`.cache/polyglot-runtime.json`, its Docker image, and the cached coding benchmarks.
Dependencies must be prepared before starting: the worker runs with Hugging Face offline mode.
These prerequisites are checked again by the training and evaluation tools.

## Use it

- **Home** shows the learning state, current activity, GPU use and measured quality.
- **Activity** shows the current stage log and complete saved checkpoints.
- One main button shows **Start** when idle or paused, and **Stop** while learning.
  **Start** resumes saved work. A stopped continuous loop shows **Resume**.
  A finished experiment shows **Start new session**; it does not reset old results.
- **Pause** requests a checkpoint and releases the GPU. Wait for **paused**.
- **Stop** stops learning after confirmation. A continuous loop can resume later.
  It stays stopped after login until you select **Resume**. Use Pause for a break.
- **Results** holds past comparisons. **Settings** holds setup, GPU power and login options.

Closing the app leaves the worker running. Saved Pause and Stop remain in effect after login.
Training Studio adds a task named `TrainingStudio-…` only when a worker needs to start.
Existing workers continue to use their existing controls. One global GPU lock prevents
simultaneous training runs; pause the active experiment before starting another.

## Appearance

The **Appearance** selector in Settings defaults to **System**, which follows Windows app
light/dark settings. Choose **Light** or **Dark** to override it for Studio only.
The choice is saved locally and applies without restarting. The native Windows Mica
backdrop and translucent cards add subtle depth. Windows controls the backdrop;
turning off transparency or using a contrast theme gives solid surfaces. Status polling keeps
the experiment selector and controls steady; explicit actions still prevent duplicate clicks.

## Startup preferences

Open **Settings → Startup** and select **Save startup preferences** after making changes.

- **Resume learning automatically after Windows login** is on by default. It adds login
  and recovery triggers. Saved Pause, Stop and completed experiments stay respected.
- **Open Training Studio after Windows login** is off by default. Turn it on to show
  the dashboard when you sign in. This switch alone does not start learning.

Turn both off to open Studio yourself and press **Start** manually.
Disabling automatic learning removes login, timer and failure-retry starts. Changing
these preferences does not stop work already running; use **Pause** for that.
Settings stay in `%LOCALAPPDATA%\TrainingStudio\startup.json`.
The controls apply to Studio workers, not legacy PowerShell tasks or unrelated apps.

## GPU power

Open **Settings → GPU power**, select the NVIDIA GPU, and choose **Enable power control** once.
Windows asks for administrator permission for this setup. Then use the slider or
5% preset buttons and select **Apply training limit**. The percentage uses the GPU's
default wattage, not current utilization. The driver supplies the minimum; presets
above 100% or below that minimum are unavailable.

Studio shows both the pending choice and the actual limit read back from NVIDIA.
It saves the selected limit for the current connection and experiment on this PC.
Changing the slider alone does not change power. **Restore 100%** also saves 100%
as the training setting.

- **Pause** restores 100% while the checkpoint saves. Wait for **paused** before gaming.
- **Start** reapplies the saved limit when the worker becomes active, within its
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

Open **Settings → New session**. This release supports **Qwen3.5-4B BF16 LoRA** on a 16 GB NVIDIA GPU.
Other architectures are not supported by the wizard yet.

Choose a unique name and select **Create session**. Select **Start**
when ready. The loop has no time limit. Pause saves progress and releases the GPU;
Resume continues it. Stop keeps learning off until you explicitly resume.
Login recovery respects both controls and never creates a replacement session.

Each cycle collects complete programs from fresh source problems, checks them in isolated
containers, and keeps only passing answers. Training uses balanced Python, Go, TypeScript,
Rust and Dart examples plus earlier verified answers. The base weights stay frozen;
small LoRA adapters receive at most 128 optimizer steps and one epoch per cycle.

The 0.1.4 source preview also trains successful public-feedback corrections and executed REPL
repair lessons. A separate matched REPL pilot runs before and after each training block.
A candidate must gain a direct-validation or REPL pass without losing an earlier pass in either,
then preserve earlier passes in a matched 790-task comparison before it replaces the retained
model. Periodic full comparisons also check candidates that did not pass the initial gate.
Rejected candidates are removed and collection continues. Existing completed experiments
stay available in the picker. Integrity failures still need attention; temporary worker failures
retry with a delay.

## Read the result

Home shows the current editable source with a copy button, completed first attempts,
repairs, and verified REPL training rows. The compact toolbar keeps Pause and Stop visible.
The Windows 11 dashboard uses Mica and translucent panels, and respects the system
transparency setting. Detailed comparisons are in Results.

**Not yet measured** means matched tests are not complete. Partial counts show progress.
**Retained model unchanged** means no candidate replaced the current model.
**More verified examples needed** means collection continues before training.
For a rejected candidate, its measured gains and losses remain visible. A rejection does not stop the loop.

The 790 function tests cover five languages. These public tests may occur in model pretraining.
They are reused for selection and do not establish general coding or agent quality. Fresh
validation problems are separated from training by problem family, including translations.

See the [learning method](https://github.com/M1chlCZ/local-coding-assistant/blob/studio-v0.1.3/research/verified-learning.md) for sources and limitations.

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

On a connected Windows profile, `--check-ui <output>` verifies steady controls during
a background read, a command queued behind that read, and text contrast in both themes.
It renders every tab in Light, Dark and System modes without changing saved preferences
or issuing training controls.

## Teacher examples

New sessions can mix published NVIDIA reference functions with verified replay.
Studio checks the reference tests locally and keeps validation answers separate.
The first source is Python; the full regression check still covers five languages.
The dashboard shows the teaching method, example counts and accepted updates.
See [the pilot and its limits](../research/teacher-training.md).

## Execution learning preview

New sessions in 0.1.4 use [execution learning](../research/execution-learning.md).
Home shows the five-stage cycle, current task, first-attempt passes, successful corrections,
executed REPL training rows and the latest rejection reasons. Long sessions display elapsed days.
REPL skills have their own before/after result; training activity does not count as improvement.

The Windows 11 Fluent theme supplies the Mica window backdrop. Cards, selected stages and
inset surfaces have translucent fills with fully opaque text. Windows transparency preferences
and high-contrast settings take precedence. Light, Dark and System themes remain available.
