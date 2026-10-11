# Continuous learning method

This is a local experiment for Qwen3.5-4B. It uses execution-filtered self-training
and small LoRA updates. It is not a reproduction of a published benchmark result.

## Sources

- [SelfCodeAlign](https://github.com/bigcode-project/selfcodealign/tree/045354e1f60ed01b0652ff9716359bb73d246740)
  publishes its code, data, models and evaluations. Its
  [StarCoder2 recipe](https://github.com/bigcode-project/selfcodealign/blob/045354e1f60ed01b0652ff9716359bb73d246740/README-SC2INST.md)
  generates several answers and filters them by execution. It uses ten samples at
  temperature 0.7, followed by full fine-tuning on an A100 with 80 GB memory.
  Our smaller loop borrows the filtering method, not its hardware settings or result.
- [Unsloth's Qwen3.5 example](https://github.com/unslothai/notebooks/blob/8daba2b3ee978ac22c3573f1ff1728683a1a6e3e/nb/Qwen3_5_(4B)_Vision.ipynb)
  demonstrates 16-bit LoRA with gradient checkpointing on this model family.
  It is a vision example, not evidence that our coding dataset improves the model.
  We use the installed Unsloth backend with the text model and assistant-only loss.

## Each cycle

1. Reserve fresh training and validation problems from the pinned CodeContests source.
   One problem and all of its language versions stay in the same split.
2. Generate complete programs in five languages. Try one greedy answer, a correction
   if public examples fail, then up to two samples at temperature 0.7 if needed.
   Source test answers are never supplied as correction feedback.
3. Keep only programs that pass the registered source tests in isolated containers.
   Recheck earlier passing training answers and mix them with new answers. Balance
   the five languages. Exclude examples that do not fit the context without truncation.
4. Evaluate the retained model on fixed and fresh validation problems. Use one greedy
   answer per problem, without corrections or extra samples.
5. Train a candidate from the retained LoRA adapter, or the original base on the first
   cycle. Use at most ten optimizer steps, learning rate 0.000005, rank 8, batch size
   1 with eight accumulation steps, and an 8-bit AdamW optimizer. Only answer tokens
   receive loss. The original BF16 model weights stay frozen. The run records the
   number of trainable parameters and verifies that only LoRA weights can change.
6. Evaluate the candidate under the same conditions. Before acceptance, require a
   validation gain with no lost passes and a full matched 790-task comparison with
   no lost passes. Also run the full comparison every fifth cycle with a candidate.
7. Keep an accepted adapter. Discard rejected candidate weights, retain its report,
   and continue with fresh problems. Rejection does not end the loop.

There is no wall-clock limit. Pause saves progress and releases the GPU. Resume uses
the saved optimizer checkpoint within a block. A new block starts a fresh optimizer
from the retained adapter. Stop ends the session. Temporary failures retry with a
delay; integrity failures require attention.

## What this can and cannot show

Passing source tests is stronger than copying unchecked generated code. It does not
prove a program is correct for every input. Public sources may overlap pretraining.
The full benchmark is reused for model selection, so its scores are development
results. Separate untouched tests are still needed for a final quality claim.

LoRA is real weight training: gradients update small adapter matrices attached to
selected attention and feed-forward projections. It does not retrain every base
weight. Full fine-tuning needs much more optimizer memory. A published successful
method does not guarantee progress on a different model, dataset or memory budget.

The ten-step blocks, replay balance, sample limit and promotion rules are local
design choices. Their value must be measured here. More runtime alone is not evidence
of better code. Keep both accepted and rejected results visible.
