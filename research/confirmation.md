# Reserved coding comparison

On 3 October 2026, the PC checked ten public repair tasks reserved before the replay research session.
These tasks supplied no training examples or candidate selection results.
The comparison checked all ten tasks for each model, with no early rejection.

| Qwen3-4B variant | Passing repairs | Model answer time | Model calls |
| --- | ---: | ---: | ---: |
| Unchanged NF4 base | 1/10 | 466.98 s | 78 |
| Earlier accepted adapter, 16/25 development | 9/10 | 75.47 s | 20 |
| Latest accepted adapter, 18/25 development | 9/10 | 77.45 s | 20 |

Both adapters passed the same nine tasks.
The latest adapter preserved those passes, but it did not improve the confirmation score.
Its higher repeated development score does not establish broader improvement.
These results support this small Python repair harness. They do not prove general coding or agent reliability.
Overlap with model pretraining data is unknown.

## Comparison conditions

The comparison used pinned Qwen3-4B weights, NF4, greedy generation, the same REPL harness, and the same task order.
Each task received at most eight model calls, 4,096 output tokens, and 120 seconds.
The server allowed at most 1,024 output tokens per call.
The comparison checked reference repairs and incomplete solutions in containers before model evaluation.
Reference repairs supplied no model answers or training targets.
The teacher, trainer, and chat server stayed closed during comparison.

The [aggregate report](../reports/confirmation-2026-10-03/summary.json) records model hashes, source hashes, task outcomes, and settings.
Raw model traces and the task registry remain in ignored local storage.
This confirmation set is now used. Future model selection needs new confirmation tasks.
Training must not target the remaining failed confirmation task.

## Session recovery

The research session collected all 309 prepared public tasks. Container checks accepted 299 teacher repairs.
The next batch contained no training tasks.
The previous collector advanced to export without saving a teacher report, which caused a missing-file error.
The corrected collector saves an empty report and finishes with `curriculum_exhausted`.
The exporter also recovers this condition after interruption.
Accepted checkpoints, elapsed time, and the original active time limit stay intact.

The Windows worker now reports native failure codes correctly.
The controls also accept a worker that completes during startup and reject attempts to restart finished sessions.
Regression checks cover empty collection, interrupted export, missing nonempty reports, native failures, and startup completion.
