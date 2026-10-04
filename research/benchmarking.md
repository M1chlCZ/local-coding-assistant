# Coding benchmarks

Keep training checks and benchmark results separate.
Training loss does not measure coding correctness.

| Check | Purpose | Current evidence |
| --- | --- | --- |
| 25 repeated development repairs | Select training candidates | Accepted adapter: 18/25 |
| Ten reserved public repairs | Compare the base and accepted adapters with the repair harness | Base: 1/10. Both accepted adapters: 9/10 |
| Full HumanEval | Measure direct Python generation on 164 public tasks | Base: 118/164. Accepted adapter: 121/164 |
| 20 fresh reserved repairs per experiment | Check the final adapter against the starting adapter and base | Automatic comparison after each experiment |

## Automatic schedule

Continuous mode owns this schedule on the PC.
The Mac chat does not start each comparison.

| When | Check | Result use |
| --- | --- | --- |
| After each training round | Development repairs | Accept only complete results with more passes and no lost tasks |
| After each experiment, with a six-active-hour limit | 20 fresh reserved repairs | Compare the base, starting adapter, and final adapter. Return to the starting adapter after a regression |
| After that fresh comparison, before the next experiment | Full 164-task HumanEval comparison | Report coding outcomes and speed for the adapter that continues |

HumanEval runs again only when the accepted weights change.
Unchanged weights reuse a complete report with the same benchmark settings and source hashes.
Interrupted comparisons retain completed tasks and continue after Resume or Windows login.
Gaming Pause stops the comparison and releases its model server.
The worker continues with fresh training data after the comparison finishes.

The Windows panel shows the schedule, current benchmark progress, and last complete result.
Its time estimate counts active experiment time. Pause and other benchmark work extend the calendar interval.
An experiment can finish early because of its task or storage limit.

HumanEval reports trends. It does not select checkpoints or supply training targets.
Repeated use can bias research decisions, even without direct training on its tasks.
The fresh reserved repairs remain the separate regression check.

The [reserved comparison](confirmation.md) supports a small Python repair workflow.
The later adapter did not improve its confirmation score.
The repeated development score does not establish broad improvement.

## HumanEval comparison

The CUDA comparison finished on 4 October 2026.

| Qwen3-4B variant | Passing tasks | Pass rate | Median answer time |
| --- | ---: | ---: | ---: |
| Unchanged NF4 base | 118/164 | 72.0% | 3.41 s |
| Accepted adapter, 18/25 repair development | 121/164 | 73.8% | 5.57 s |

The adapter gained 14 tasks and lost 11, for a net gain of three tasks.
It also took longer to answer.
This mixed result does not establish broad coding improvement without regressions.
The [aggregate report](../reports/student-humaneval-2026-10-04/summary.json) includes hashes, settings, and every task outcome.

The comparison uses the same pinned Qwen3-4B model, NF4, prompt, and task order.
It enables or disables the accepted adapter in the same model server.
Each task receives one greedy answer, with at most 1,024 output tokens.
The comparison disables thinking, repair tools, and retries.
Only the task prompt and fixed system instruction go to the model.
Test code and reference answers stay outside model requests.
All 164 reference solutions must pass isolated container checks before model evaluation.
The script saves a comparison score after both models complete every task.

The report includes passing tasks, gained and lost tasks, answer time, output tokens, and model hashes.
Request speed includes prompt processing and HTTP overhead.
It does not measure isolated decode speed.

This comparison uses the original HumanEval tests.
The execution environment provides the Python standard library and disables site packages.
It does not use [HumanEval+](https://github.com/evalplus/evalplus), which adds stronger tests.
Public benchmark overlap with pretraining and public training sources is unknown.
Keep these benchmark tasks out of future training and checkpoint selection.
Use fresh repository tasks to measure broader agent usefulness.

## Run in the Linux project folder

First, select Pause in the Windows learning controls.
Wait for Paused, then close the chat server.
Replace the example adapter path with a complete saved checkpoint.

```bash
.cache/rlm-env/bin/python research/student_benchmark.py \
  --adapter .cache/learning/research/round-008/adapter/checkpoint-5 \
  --output .cache/benchmarks/student-humaneval \
  --paused-controller .cache/learning/continuous
```

The command saves each completed task.
Press Ctrl+C to interrupt the benchmark and release its model server.
Run the same command to continue an interrupted comparison.
A different adapter, dataset, or benchmark source requires a separate output folder.
Select Resume when the comparison finishes.
The handoff closes only the identified paused worker to release its ownership lock.
The saved session and checkpoints remain intact.

Raw answers, logs, and local paths stay under ignored `.cache/` storage.
Publish only a reviewed aggregate report.

## CPU checks

```bash
python3 -m unittest test_student_benchmark test_scheduled_benchmark
```

These checks reject incomplete comparisons and mismatched saved progress.
They also detect individual regressions when the total score increases.
The schedule checks cover audit ordering, rollback selection, resumable output, and durable Pause.
