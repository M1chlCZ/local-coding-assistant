# Coding benchmarks

Keep training checks and benchmark results separate.
Training loss does not measure coding correctness.

| Check | Purpose | Current evidence |
| --- | --- | --- |
| 25 repeated development repairs | Select training candidates | Accepted adapter: 18/25 |
| Ten reserved public repairs | Compare the base and accepted adapters with the repair harness | Base: 1/10. Both accepted adapters: 9/10 |
| Full HumanEval | Measure direct Python generation on 164 public tasks | Matched comparison in progress |
| 20 fresh reserved repairs per experiment | Check the final adapter against the starting adapter and base | Automatic comparison after each experiment |

The [reserved comparison](confirmation.md) supports a small Python repair workflow.
The later adapter did not improve its confirmation score.
The repeated development score does not establish broad improvement.

## HumanEval comparison

The comparison uses the same pinned Qwen3-4B model, NF4, prompt, and task order.
It enables or disables the accepted adapter in the same model server.
Each task receives one greedy answer, with at most 1,024 output tokens.
Thinking, repair tools, and retries are disabled.
All 164 reference solutions must pass isolated container checks before model evaluation.
Every model must complete every task before a comparison score is saved.

The report includes passing tasks, gained and lost tasks, answer time, output tokens, and model hashes.
Request speed includes prompt processing and HTTP overhead.
It does not measure isolated decode speed.

This comparison uses the original HumanEval tests.
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
Run the same command to continue an interrupted comparison.
A different adapter, dataset, or benchmark source requires a separate output folder.
Select Resume when the comparison finishes.
The handoff closes only the identified paused worker to release its ownership lock.
The saved session and checkpoints remain intact.

Raw answers, logs, and local paths stay under ignored `.cache/` storage.
Publish only a reviewed aggregate report.

## CPU checks

```bash
python3 -m unittest test_student_benchmark
```

These checks reject incomplete comparisons and mismatched saved progress.
They also detect individual regressions when the total score increases.
