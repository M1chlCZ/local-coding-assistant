# Teacher-backed coding pilot

The old loop learns from answers that the same model already solved. A small
pool of these answers can change output style without improving coding. It also
trains on whole programs, while HumanEval and MultiPL-E measure functions.
These are plausible causes of poor transfer, not a proven diagnosis of every
failed trial. Greedy GPU results can also vary between runs.

## What changed

New Studio sessions enable optional teacher examples by default. Each cycle checks
up to 128 new reference functions from NVIDIA OpenCodeInstruct. The download is pinned to a
commit and SHA-256, and uses one 128 MiB shard, not the full dataset.

Each reference must pass at least five supplied assertions in a network-free,
resource-limited container. The same tests must reject an empty solution.
Publisher scores alone are not enough. Overlong examples are excluded rather
than cut in half. Teacher answers receive training loss; prompts do not.

Twenty different teacher functions join the existing multilingual validation
tasks. Function-name families stay in one split, and text overlap checks reject
near copies across splits. Benchmark function names and 13-word overlaps are
excluded. This cannot prove that there is no semantic or pretraining overlap.
Validation reference answers and assertions never enter the model prompt or
training data. All source data and results stay in ignored local cache folders.

Teacher functions are mixed with the existing verified replay in Python, Go,
TypeScript, Rust and Dart. **The new teacher source is Python only.** Synthetic
tests can have mistakes; passing them is evidence, not a correctness guarantee.
Do not claim improvements in other languages without measuring them.

The current [execution recipe](execution-learning.md) starts at 5e-6, with one epoch
and at most 128 steps, and caps language pools. Existing `teacher_mix` sessions
keep their earlier one-epoch trial at 2e-5, capped at 256 optimizer steps. This is a local starting hypothesis, not a reproduced NVIDIA
recipe. Old self-training failures do not set this new recipe's learning rate.
Subsequent validation or full-audit losses reduce the next teacher rate, with a
floor of 5e-6. A full audit takes priority when available. Started blocks
always keep their saved optimizer and recipe on Resume.

The existing worker, checkpoints, Pause, Resume, Stop and login behavior remain.
Every promising candidate must pass the complete matched five-language audit
before promotion. The no-lost-task rule remains conservative. Audit results are
reused development evidence; they are not a fresh estimate of general quality.
After three teacher updates without a full audit, the next candidate receives
the full comparison even if validation rejected it. This supplies diagnostic
evidence; it does not relax promotion rules. Test-only imports run in a separate
namespace so correct import styles work and missing candidate imports still fail.
If the pinned source runs out, Studio reports that it needs another reviewed
source. It does not silently recycle the same batch forever.

## Research and attribution

- [NVIDIA OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)
  provides reference answers, execution outcomes and tests for supervised coding
  training. NVIDIA; Ahmad et al., 2025; [paper](https://arxiv.org/abs/2504.04030).
  Data license: CC BY 4.0. Keep this attribution with derived data distributions.
- [MultiPL-T](https://github.com/nuprl/MultiPL-T) reports improvements from tested
  translated examples in lower-resource languages. It supports the approach of
  adding checked external examples; this pilot does not reproduce its results.
- [OpenCoder](https://github.com/OpenCoder-llm/OpenCoder-llm) releases its data and
  supervised training pipeline. Its training scale is far beyond this local
  pilot, so its published scores are not a promise for a 16 GB PC.

Next evidence needed: complete the matched pilot, inspect per-language gains
and losses, and compare with an unchanged-base repeat when a delta is small.
Do not increase trial length merely because GPU time is available.
