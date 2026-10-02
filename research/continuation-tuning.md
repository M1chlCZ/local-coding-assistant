# Continuation tuning

The accepted Qwen3-4B adapter passed 9/15 authored repairs and 7/10 public coding tasks.
Three later updates at `5e-5` scored 4/15, 4/15, and 5/15.
The worker rejected every update and retained the accepted adapter.

The updates used 150 examples. Later rounds repeated the same problems with new labels.
The exporter removed repeated examples, so those rounds supplied no new training data.
Several failed responses repeatedly inspected the context instead of returning a repair.

## Measured trials

The first trials changed one setting at a time on the same fixed dataset.
At `5e-6`, the 38-step update failed the first regression check.
The five-step update passed two regression checks but failed the third.
These partial checks rejected the candidates. They do not measure complete development scores.

The next trial used 32 fresh NVIDIA OpenCodeInstruct tasks and replay from the first 32 tasks.
The teacher passed 31/32 fresh tasks and 61/64 tasks overall.
The student used 20 optimizer steps at `5e-6`, NF4, rank-8 QLoRA, and a one-pass ceiling.

| Adapter | Authored repairs | Public coding tasks | Decision |
| --- | ---: | ---: | --- |
| Accepted adapter | 9/15 | 7/10 | Retained |
| Step 20 | 8/15 | 8/10 | Rejected: lost an older passing repair |
| Step 10 | 9/15 | 7/10 | Rejected: no improvement |
| Step 15 | 3/4 checked | Not checked | Rejected at the first lost prior pass |

The [aggregate report](../reports/continuation-tuning-2026-10-02/summary.json) records settings, hashes, memory use, and selection results.
Step 10 belongs to the 20-step run. It does not represent a separate ten-step training schedule.
All trials retained the original accepted weights without changes.
The original holdout and the other 35 public development tasks remain unused.
Private chat history supplies no training examples.
These small, repeated development sets do not establish general coding reliability.

## Changes to the worker

Updates from an accepted adapter now use at most 20 steps at `5e-6`.
The evaluator checks retained checkpoints instead of selecting the final step automatically.
An optional, fixed development registry adds public checks to the acceptance rule.
The worker stops before another update when a dataset adds no new examples.

These controls reduce wasted updates and preserve previous passing tasks.
They do not establish an improvement in the trained model.
All datasets and adapters stay in ignored local folders.

See [the public data source and license](public-data.md) and [Windows controls](learning-session.md).
