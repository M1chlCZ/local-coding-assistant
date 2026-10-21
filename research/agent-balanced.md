# Bounded balanced recovery trial

The completed trial did **not** meet the development gate. Both models solved
one of 16 tasks, but the candidate gained text splicing and lost token-bucket
handling. Neither model completed a qualifying ordinary tested REPL repair or a
genuine recursive repair. Full Collect remains paused. The candidate was not
selected, and the eight reserved heldouts and new 790-task audit were not run.

The [aggregate report](../reports/agent-balanced-20261021.json) records individual
changes, verification hashes, training provenance and the interrupted first run.
Raw trajectories, datasets, checkpoints and recovery records remain private.

## Fixed method and completed results

This was one fresh Qwen3.5-4B BF16 adapter and optimizer: rank 8, learning rate
0.000005, 96 updates, accumulation 8, maximum length 4,096 and seed 3407.
No rejected adapter initialized it. Of 64 deterministically selected training
families, 63 complete ordinary/child pairs fit the frozen whole-trajectory token
rule. The 126 executed trajectories supplied 1,090 action rows; 86 failed proposals
were masked from targets. Another 1,090 rows explicitly repeated 150 distinct
verified multilingual anchors. This was row balance, not token balance or 2,180
distinct problems. All 2,180 prepared rows were encoded without further exclusion.

Both arms used the same frozen helpers, prompts, tokenizer context policy and
shared limits: depth 2, 12 model calls, 8,192 output tokens and 360 seconds.
Development comprised 16 new public tasks, eight ordinary and eight recursive,
requiring two-file repairs without supplied test files. Fifteen general-code
retention checks were also completed in each arm.

| Measure | Retained BASE | Candidate |
| --- | ---: | ---: |
| Functional development repairs | 1/16 | 1/16 |
| Ordinary tested REPL repairs | 0/8 | 0/8 |
| Genuine recursive repairs | 0/8 | 0/8 |
| Retention passes | 0/15 | 0/15 |
| Returned root patch objects | 12 | 3 |
| Actual child checks | 0 | 1, failed |
| Observed completed `finish()` operations | 0 | 0 |

One gained family is below the required two; the functional loss and missing
ordinary/recursive gains also independently reject the candidate. Zero baseline
retention passes cannot demonstrate preservation of general coding ability.
Training completion, more assertions or more delegation calls are not learning
gains.

## Evidence review and interface limitation

Independent verification reconstructed all 303 development generation contexts
across 32 trajectories. There were no compactions, context overflows or recorded
development infrastructure errors. Initial prompts matched for every paired task.
All 111 configured input hashes were checked. All 15 returned root patches were
regraded without discrepancies. All five child returns were budget-exhaustion
errors, so there were no valid child patches to regrade.

Review of every executed check found meaningful successful parent assertions on
the BASE token-bucket implementation and candidate text-splicing implementation.
Neither returned final patch matched the recorded workspace hashes, and neither
completed `finish()`. They therefore do not satisfy the frozen tested-REPL rule.
The only child test imported the real undo implementation and ran meaningful
assertions, but failed. It did not repair, pass a new check and finish. Other
children mainly inspected files before exhausting the shared budget.

A shared interface problem limits interpretation: `read(path)` returns a string,
while the worker executes cells with Python `exec` and does not display expression
values. Bare `read(...)` calls therefore produced `REPL output: No output` in every
task in both arms. Repeated attempts to inspect the supporting bridge consumed
actions, and many returned callers used its arguments incorrectly. This is an
observed mismatch between model behavior and the interface, not a context overflow.
The saved sources were not changed and no additional recipe or model evaluation
was launched. These results reject this candidate under this interface; they do
not establish how training would perform after an interface repair.

## Host interruption and recovery

The original run stopped during its fifth baseline task when Windows shut down
uncleanly and Docker canceled execution. Four partial rows existed, but training
had not started. The physical cause remains unknown. Two subsequent Resume
attempts were rejected by the stage guard before inference.

The reviewed continuation inherited only the three completed preparation/data
stages with exact hashes. It preserved the old snapshot and started a complete
new baseline, then the original single recipe and candidate evaluation. No partial
model result was reused. The continuation completed all stages with no controller
failures; the experiment's overall history still includes the host interruption.

All evidence and checkpoints are retained. The finite trial is complete, the full
loop remains paused, and no further training recipe is authorized by this result.
