# Matched evaluation after repairing file-read visibility

The read repair works, but the unchanged learned adapter fails the required
development gate. In a new complete comparison, retained BASE solves 7 of 16
tasks and the candidate solves 1. The candidate loses six functional repairs and
both BASE tested ordinary REPL successes. Neither model completes a qualifying
recursive repair. Full Collect remains paused; no further training was run.

The [aggregate report](../reports/agent-read-20261021.json) records all individual
losses and verification hashes. Raw trajectories, datasets and checkpoints remain
private. This follows the [balanced trial](agent-balanced.md), whose original
results and interrupted host history are preserved.

## Shared repair and comparison

The worker executes cells with Python `exec`, which discards bare expression
return values. `read(path)` previously returned source without displaying it.
It now displays and returns the exact source. Internal reads used by `show`,
`delegate` and `finish` remain quiet, and the typed dispatcher no longer prints a
read twice. Existing path restrictions, source validation and output limits remain.

Fourteen preflight tests passed on the target machine, including actual parent
and child sandbox reads, return-value preservation and rejection beyond the
existing output cap. Source output was also observed in the next model prompt
during evaluation. These establish an interface repair, not learned improvement.

Both arms received the repaired helper, identical initial task prompts, the same
BF16 greedy runtime, exact tokenizer budgeting and shared limits: depth 2,
12 model calls, 8,192 output tokens and 360 seconds per task. Each request contained
one conversation; the server batch capacity was two. The candidate weights are
exactly the completed balanced adapter, with no additional gradient updates.
Both arms were evaluated anew; no previous model outputs were reused.

The 16 public multi-file tasks are **already-exposed development tasks**, eight
ordinary and eight recursive, with requirements and examples but no supplied
test files. They are not fresh heldout evidence. Each arm also completed the same
15 retention checks.

| Measure | Retained BASE | Unchanged candidate |
| --- | ---: | ---: |
| Functional development repairs | 7/16 | 1/16 |
| Ordinary tested REPL repairs | 2/8 | 0/8 |
| Genuine recursive repairs | 0/8 | 0/8 |
| Retention passes | 0/15 | 0/15 |
| Returned root patch objects | 12 | 2 |
| Actual child checks | 0 | 9 |
| Successful child assertion checks | 0 | 3 |

There are no gained development families. The six functional losses are alias
following, version comparison, header redaction, text splicing, clockwise
rotation and cooldown event acceptance. Alias following and version comparison
also lose qualifying tested REPL success. Token-bucket handling is the only
functional pass retained by the candidate; it has no parent test or completed
`finish()` and therefore is not a tested REPL or recursive success.

## Independent review

All 113 configured input hashes and 312 frozen public source files were verified.
The complete 32 trajectories have matching initial prompts and weight, evaluation
and runtime bindings. Independent tokenizer replay reconstructed all 305
development generation contexts, with no compaction or context exhaustion.
All 14 returned root patches were regraded with no discrepancy. All seven child
returns were budget-exhaustion error strings, leaving no valid child patch to
regrade. The comparison completed its three guarded stages once, with no
controller failure or model retry.

Every recorded assertion check and returned root patch was inspected. BASE's two
ordinary tested successes call real repository functions and have matching tested
source hashes and completed finish receipts. Its other functional passes do not
satisfy that complete checked-finish chain. A third BASE finish fails functional
grading and does not count as a solve.

Candidate children performed real work: the rotation child passed four assertions
after writing source, then added an incorrect expectation and failed; the event
acceptance child passed two meaningful assertions twice after repairing source.
Neither finished and returned a patch. All seven child calls exhausted the shared
budget, and parsing those error strings caused parent JSON errors. No completed
child repair was applied and successfully tested in the parent. Passing isolated
child checks is insufficient for genuine recursive success.

This is a failed learning gate under the repaired interface, with no recorded
development infrastructure failure. The earlier training history still includes
the reviewed Windows shutdown. An initial dashboard check also ran before its
Pause button loaded; a subsequent read-only UI verification passed without any
model restart. Those events remain recorded rather than being described as a
pristine overall history.

## Decision and limits

The candidate was not selected. The eight reserved never-model-evaluated heldouts
remain unchanged and unused. The new matched 790-task audit was not started
because development already failed. Both arms scoring zero on the 15 retention
checks does not demonstrate general-code retention.

The result rejects this adapter under this interface; it does not determine why
the training recipe failed or test another recipe. Read visibility improvements,
successful helper tests and additional child activity do not authorize Collect.
All checkpoints and evidence are preserved, and the full loop remains paused.
