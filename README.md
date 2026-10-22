# <img src="desktop/TrainingStudio/Assets/logo-icon.png" width="48" height="48" alt="Training Studio logo"> Training Studio

[![Support me on Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/m1chl)
<a href="https://buymeacoffee.com/xdujifyzvt"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy me a coffee" height="30"></a>

Train a coding model on your PC. Training Studio runs fine-tuning through WSL and CUDA, shows the results, and lets you pause training to free the GPU for gaming.

This is a research prototype for 16 GB NVIDIA GPUs. It needs a prepared WSL backend. More training does not guarantee better code.

[Download Training Studio](https://github.com/M1chlCZ/local-coding-assistant/releases/tag/studio-v0.1.3) · [Installation guide](desktop/README.md) · [Learning method](research/verified-learning.md)

![Training Studio running on Windows 11 with muted dark translucent panels](reports/training-studio-windows11.png)

*Windows 11, version 0.1.4-preview.5, in Dark mode during the recovery baseline. The tinted Mica background, translucent slate panels and darker code areas reduce bright areas. This is an earlier capture. The capture after the comparison returned blank frames. The full loop is paused while later finite experiments are evaluated.*

## Windows training app

Version 0.1.3 adds a simpler interface and a continuous training loop. It trains Qwen3.5-4B on Python, Go, TypeScript, Rust and Dart examples. Model files stay on your PC. So does private data.

The app shows live activity, GPU use, saved checkpoints and coding results. Here is where to find them:

| Control or page | What it does |
| --- | --- |
| Start / Stop | Starts or stops the session. |
| Pause | Saves progress and releases the GPU for gaming. |
| Home | Shows whether the model improves. |
| Activity | Shows progress and errors. |
| Results | Shows past comparisons. |
| Settings | Contains startup preferences, GPU power and appearance. |

The interface follows the Windows theme by default, with a subtle Mica background. Dark mode is also available.

The loop collects answers that pass code tests, trains a small adapter, then compares it with the retained model. Rejected updates never replace the retained model. The loop continues after a rejection.

The 0.1.4 source preview adds [execution learning](research/execution-learning.md): successful correction conversations, executed REPL repair lessons, a capped multilingual training mix, and matched recursive-agent checks before accepting an update. Its Windows 11 dashboard shows the current task, repair counts and rejection reasons, with Mica and translucent panels. Existing experiments and checkpoints stay separate.

[Download 0.1.3](https://github.com/M1chlCZ/local-coding-assistant/releases/tag/studio-v0.1.3) · [Release notes](releases/0.1.3.md) · [Installation guide](desktop/README.md) · [Learning method](research/verified-learning.md)

## What the recent experiments found

The full learning loop remains paused. These experiments measure different parts of the setup; a development gain alone does not qualify an adapter for a restart.

Both predeclared parent/child training settings completed with 0/8 repairs and 0/4 recursive repairs, matching the base model. Neither qualified. See the [measured pilot results](reports/agent-curriculum-pilot-20261019.json).

A separate corrective-imitation experiment improved from 0/10 to 2/10 repairs. Ordinary REPL repairs rose from 0/5 to 2/5; recursive repairs stayed at 0/5. Independent regrading confirmed both gains. The recursive requirement failed, so no adapter was promoted. Held-out confirmation and the full general-code audit were not run. See the [verified recovery result](reports/agent-recovery-pilot-20261019.json) and [failure diagnosis](research/execution-learning.md#completed-corrective-imitation-comparison).

The [child-tested experiment](research/execution-learning.md#execution-tested-recursive-distillation) improved development repairs from 2/10 to 6/10 under matched instructions. Genuine recursive repairs rose from 0/5 to 3/5, while ordinary REPL repairs stayed at 2/5. All returned patches were independently regraded. Confirmation failed. The selected candidate scored 0/5 against the base model's 0/5 on five untouched repairs, so it was rejected. No full audit or restart followed. See the [complete measured outcome](reports/agent-tested-development-20261019.json).

The [multi-file trial](reports/agent-multifile-trial-20261019.json) was also rejected. Development repairs fell from 1/15 to 0/15; ordinary REPL repairs stayed at 0/8 and genuine recursive repairs at 0/7. Its lessons derived checks from specifications without supplied test files, but the candidate returned no final patches. Independent verification regraded both baseline patches and confirmed the regression. The eight fresh confirmation tasks remained unused. No general-code audit or full-loop restart followed.

The [short-action recovery trial](research/execution-learning.md#short-repl-actions-and-recovery-imitation) reached a [fresh matched comparison](reports/agent-context-comparison-20261020.json) after a context-overflow repair. It regressed too. Functional repairs fell from 5/15 to 0/15, strictly tested ordinary REPL repairs from 2/8 to 0/8, and genuine recursive repairs stayed at 0/7. All 277 generations and 12 returned root/child patches were independently checked. The candidate was rejected. The eight reserved confirmation tasks remain unused, and the full loop stays paused.

The Qwen evaluation server supports optional [schema-constrained JSON output](research/structured-output.md) through XGrammar. Constraints are off by default. In a completed four-task comparison with the same retained BASE weights and identical prompts, both modes produced valid JSON on 4/4 tasks. Functional repairs fell from 2/4 to 1/4 with constraints. All eight patches were independently regraded. This result establishes no REPL or recursive learning gain.

The [DeepSeek source review](research/deepseek-protocol.md) led to an isolated explicit-action experiment and a fix for schema-property ordering, so an action name can precede its arguments. Its 18-run comparison used six public canaries. Fenced REPL repaired 3/6; both typed variants repaired 0/6. All three had zero genuine recursive completions. Constraints reduced malformed JSON but did not improve task solving. A separate audit regraded every returned root/child patch and replayed all 159 actual contexts after automatic verification exposed a mutable receipt bug. That bug is fixed in public code. No learned gain or Collect restart is claimed.

## Coding chat and agent launcher

The Windows launcher runs English coding chat and agent sessions on a 16 GB NVIDIA GPU. It uses llama.cpp through CUDA, with CPU offloading for larger models. The system prompt requests English. The weights remain multilingual.

The default is Qwen3.8-27B with pinned Unsloth UD-Q4_K_M weights. This is also a research prototype. Setup downloads the model and runtime from their upstream repositories. Our pruned Qwen3.6 derivative is available through the [experiment recipe](research/recipe.md), but its weights are not published yet.

### Run on Windows

You need Windows x64, Python 3.10+, a recent NVIDIA driver, approximately 32 GB RAM and 20 GB free disk. Stop the model server before setup.

```powershell
py -3 launcher.py setup
py -3 launcher.py serve --open
```

You can also double-click `start.cmd`. Setup verifies download hashes; Ctrl+C stops the server. The default uses an 8K context and eight CPU FFN layers.

If memory runs out, decrease `--context`. For dense models, increase `--cpu-ffn`; for MoE models, increase `--cpu-moe`.

If the pruned derivative is already on your PC, use the faster preset:

```powershell
py -3 launcher.py --manifest research/manifest-reap50.json serve --open
```

### Connect an agent

Open chat on the PC at `http://127.0.0.1:8080`. For Mac access, follow the [SSH connection instructions](research/wsl.md#connect-from-the-mac).

| API setting | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8080/v1` |
| Model | `local-coding-assistant` |
| API key, if the client requires one | `local` |

The API listens on loopback. The external agent harness supplies tools and file access.

## Earlier training work

The [Qwen3.5-4B experiment](research/focused-training.md) trains on 2,456 complete Go and TypeScript repairs. It compares the unchanged model, the trained adapter and the previous student on matched coding tests. Windows Pause and Resume preserve progress. Automatic recovery respects Pause, Stop and completed experiments. The accepted model stays in place until the results receive review.

### REPL, recursion and training data

The [RLM experiment](research/rlm.md) adds a bounded Python REPL and optional model subcalls. A finite search compares instructions and recursion depth on separate development and holdout tasks. The [adapter recipe](research/adapter-training.md) runs local QLoRA from successful training traces.

The [confirmation comparison](research/confirmation.md) checked both retained 4B adapters on ten reserved public repairs. Both passed 9/10; the unchanged base passed 1/10. Later training updates did not raise that score.

The [WSL recipe](research/wsl.md) runs the complete experiment on the PC through CUDA. After setup, `start-rlm.cmd` starts the model server. The [learning session](research/learning-session.md) collects checked repairs and trains local adapters. Use `learning.cmd` for Windows Start, Pause, Resume and Stop controls. Checkpoints retain the optimiser and current step.

[Continuous mode](research/continuous-learning.md) prepares fresh checked public tasks and runs until Pause or Stop. The [multilingual curriculum](research/polyglot.md) rotates Python, Go, TypeScript, Rust and Dart, with mixed replay and separate language scores. An earlier full audit scored 473/790 for the base model and 433/790 for the RLM adapter.

The revised [balanced code trial](research/polyglot.md#balanced-code-trial) keeps verified examples in each language and compares fresh training with continued adapter training. It trains on verified source answers. It does not train on REPL actions, and no quality improvement is established for that recipe yet. The optional [student correction trial](research/polyglot.md#student-correction-trial) compares teacher answers with checked corrections to student mistakes.

Direct checks can batch up to sixteen independent prompts on one CUDA model and run two compiler checks in parallel. Long prompts split into smaller GPU batches under a 32,768-token budget, including padding and output. Batch-size changes require a fresh matched development baseline. An audit already in progress keeps its original settings.

A previously running session recovers after Windows login. Pause releases the GPU for gaming and stays saved across logins. The earlier tuned session uses 30 repair tasks, visible tests and short training candidates. Candidates receive 15 development checks before acceptance. Three rounds without improvement stop that session.

The [training evidence](research/rlm-training-evidence.md) compares published RLM methods, available weights and history-derived data. The [public-data pilot](research/public-data.md) prepares checked Python repairs from NVIDIA OpenCodeInstruct. Its initial 4B candidate passed 7/10 development tasks against the previous adapter's 5/10, while preserving its older passes.

The [Strata pilot](research/strata.md) measures a larger MoE teacher on the same GPU. The [private history tool](research/private-history.md) exports local Codex messages and asks the PC's model for task ideas. Exports, reviews and private adapters stay in the ignored `private-data/` directory. History-derived ideas need executable tests before training.

### Earlier measured results

The [benchmark guide](research/benchmarking.md) separates training checks, reserved repair tests and direct coding benchmarks. Training uses a separate Qwen3-4B student. On all 164 HumanEval tasks, its unchanged NF4 base passed 118 and the accepted adapter passed 121. The adapter gained 14 tasks and lost 11. A higher total therefore does not establish improvement without regressions.

The earlier continuous mode has a training limit. It stops after six active hours and schedules repair checks and a full HumanEval audit after each experiment. The Studio loop in 0.1.3 has no time limit. It uses the comparison schedule in the learning method. Unchanged accepted weights reuse verified results, and the Windows controls show the schedule and last complete comparison.

These measurements used an RTX 5070 Ti, Ryzen 9 9900X, 32 GB RAM and an 8K context:

| Model | Coding subset | Practical pilot | Median decode speed |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B, eight CPU FFN layers | 32/32 | 5/5 | 25 tokens/s |
| Our pruned Qwen3.6, no CPU expert offload | 29/32 | 5/5 | 183 tokens/s |
| MiMo V2.6 Distill 9B, no thinking | 20/32 | 4/5 | 117 tokens/s |

These are small samples. They do not establish broad agent reliability, and different quantisation methods prevent an isolated comparison of model architectures. The [comparison notes](research/candidates.md) record recent releases, memory use and failures.

On 15 authored REPL repairs, the previous 4B adapter passed 8/15 against its base at 0/15. Those are earlier baseline results. The [tuning notes](research/learning-session.md#expanded-baseline-result) separate them from the revised training run. The [model card](MODEL_CARD.md) describes our derivative and its limits.

## Reproduce

```powershell
py -3 test_launcher.py
py -3 test_evaluation.py
```

The [recipe](research/recipe.md) contains compression and evaluation commands. Docker isolates generated code during evaluation.

Before publishing, run `python3 check_public_data.py --history`. It checks for personal home paths, private history records and SSH tunnel hosts outside the documented example. To run it before each push, use `git config core.hooksPath .githooks`. Review credentials and screenshots separately.

Original code uses the [MIT licence](LICENSE). Upstream model licences remain separate. See [NOTICE.md](NOTICE.md) and [ATTRIBUTION.md](ATTRIBUTION.md) for sources and redistribution terms.

Bug reports and contributions are welcome. If the project helps you, consider giving it a star.
