# Research speed

The 16 GB CUDA card spends most research time on model answers, rather than adapter training.
In the measured run, training used about 12.5 minutes of six active hours.
Student development answers used about 3.2 hours.
The larger teacher also uses CPU layers because all its weights cannot fit in VRAM.

## Measured limits

Student decoding produced about 14–16 tokens per second.
GPU samples showed about 30–35% device activity, 6–8% memory-controller use, and 80–100 W power.
The configured power limit was 300 W.
One profiled request made about 490,000 CUDA kernel launches for 128 output tokens.
These results suggest serial decoding and small kernel overhead.
They do not establish a pure memory-bandwidth limit.

Tests with 1, 4, and 12 CPU threads gave no consistent speed improvement.
Compiled decoding failed or was skipped by the NF4 runtime.
A static-cache test also changed generated tokens.
The session keeps its original inference runtime, thread count, and generation settings.

## Stop rejected checks early

The optional fast rejection mode ends a candidate evaluation when no remaining task outcome can meet either retention rule.
It uses the highest possible final score and the passing tasks already lost.
It considers both strict promotion and the separate experimental branch.
Every accepted or retained candidate still needs all development checks.
Partial results record the number of completed checks and cannot become an accepted score.

Enable the option when you prepare a new session:

```powershell
.\learning.cmd -Action start -Hours 12 -Session .cache/learning/fast-research -Research -FastReject
```

The Linux option is `--fast-reject`.
Existing sessions keep their saved options and source hashes.
A source update requires a separate continuation folder with recorded provenance and the original elapsed time and limit.

On one real rejected checkpoint, this rule evaluated 20 of 25 tasks.
All 20 repeated outcomes matched the saved complete report.
Model answer time fell from 398.63 seconds to 317.97 seconds, about 20%.
This comparison uses the previous full report, rather than a new full run in parallel.
It does not measure complete session speed.

Replaying 30 saved checkpoint reports estimated that the rule could omit 239 of 750 checks.
That estimate saves 33.4% of student answer time, or about 18% of the recorded active session time.
All saved promotion decisions remained unchanged.
The [aggregate report](../reports/performance-2026-10-03/summary.json) separates live measurements from these estimates.
The repeated 25-task development set does not prove general coding quality.
Ten fresh confirmation tasks remain reserved for later matched evaluation.

## Invalid public tests

One public training reference failed intermittently because its test made independent random draws.
The session records such a public reference traceback as a failed, quarantined row.
The exporter excludes that row from training.
The reference answer never becomes a training target.
Authored reference failures and container infrastructure failures still stop the worker.
Task checks, task registry hashes, and acceptance rules stay fixed.

Sources: [Transformers inference optimization](https://huggingface.co/docs/transformers/main/llm_optims),
[NVIDIA performance diagnosis](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html).
