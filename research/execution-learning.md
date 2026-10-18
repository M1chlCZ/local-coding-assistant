# Learning from execution

Training Studio's new `execution_repair` recipe keeps the existing Qwen3.5-4B BF16 LoRA backend, isolated compiler containers and durable Pause/Resume controls. It changes what the model learns and how an update is compared.

## Why change the recipe

The earlier loop exported only the final passing program, even when collection needed a failed attempt and a correction. That loses the context needed to learn how to repair an answer. Published Python functions could also dominate the multilingual replay. A long-running loop and a falling training loss do not establish improved coding skill.

The new recipe records the original answer, up to three corrections using **public** execution feedback, and the successful correction conversation. Only the final correct assistant response receives loss; previous mistakes and test output are context. Hidden grading feedback never becomes a repair prompt. Independent sampled successes remain complete-code examples, without being mislabeled as successful corrections.

## REPL lessons and recursive evaluation

For a small rotating sample of verified training programs, Studio executes four steps in the existing isolated compiler container: inspect the task, reproduce the visible failure, apply the verified repair and retest, then return the patch. The starting point is the student's failing answer when available. A lesson without a reproducible public failure is excluded. These are **executed demonstrations**, not a claim that the model independently discovered those actions.

Each step becomes a separate training row with unique identity and final-response-only loss. Examples too long for the 4,096-token context are excluded without truncating their code. The dataset combines complete code, successful correction conversations, executed REPL demonstrations and optional locally checked reference functions. Each language contributes at most twice the size of the smallest language pool. Stored replay is retained; the cap applies to the training mix.

Before and after training, a separate pilot runs the model's actual recursive REPL agent on ten reserved program tasks across five languages. Both models receive the same tasks and budgets: eight calls, 6,144 output tokens, 240 seconds and maximum recursion depth two. Root and child calls share the budget. The model sees only the task, editable source and public tests. Final patches receive independent grading.

An update needs a new pass in direct validation or the REPL pilot, without losing any prior pass in either. It must also preserve every prior pass in the complete matched 790-task function audit. Failed or incomplete comparisons cannot promote a candidate. Results show direct validation, REPL and function audit separately. Existing sessions retain their original recipe; create a new session for execution learning.

## Research behind the choice

- [RISE: Recursive Introspection](https://arxiv.org/abs/2407.18219) studies iterative training that uses a model's unsuccessful attempts as the context for learning a correction. Its reported tasks and models differ from this local coding experiment.
- [SCoRe: Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917) identifies distribution mismatch and behavior collapse in offline correction training. It supports collecting current-model mistakes and measuring actual multi-turn behavior. This implementation is supervised adaptation, **not SCoRe or reinforcement learning**.
- [Recursive Language Models](https://arxiv.org/abs/2512.24601) motivates persistent REPL interaction and bounded model subcalls. Greater recursion depth is not itself a training signal or proof of improved code.
- [Hugging Face SFT Trainer documentation](https://huggingface.co/docs/trl/en/sft_trainer) describes completion-only and assistant-only loss. This repository's existing tokenizer masks the prompt and all earlier turns, training only the selected final response.

These sources motivate an experiment. They do not establish a quality improvement for this model. The small REPL pilot and repeated public function tests are development checks; a fresh held-out comparison is still needed for a broader claim.

## Operational behavior

All runtime data, prompts, traces and model weights stay under the ignored `.cache` directory. The original worker owns the GPU lock and performs cleanup on Pause. The REPL uses the prepared `.cache/rlm-env` environment; model serving and adapter training use `.cache/qwen35-env`. New session preparation requires both environments. No dependency or model download occurs during the offline worker run.

Home shows first-attempt passes and verified repairs from the current collection, the number of tokenized REPL training rows, current task and accepted updates. The decision card explains lost passes by suite. A training-row count is not a count of independent tasks or evidence of better quality.
