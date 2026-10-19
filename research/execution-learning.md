# Learning from execution

Training Studio keeps the Qwen3.5-4B BF16 LoRA backend, isolated compiler containers and durable Pause/Resume controls. The earlier `execution_repair` recipe and the separate parent/child curriculum below change training examples and evaluation, not the base model architecture.

## Why change the recipe

The earlier loop exported only the final passing program, even when collection needed a failed attempt and a correction. That loses the context needed to learn how to repair an answer. Published Python functions could also dominate the multilingual replay. A long-running loop and a falling training loss do not establish improved coding skill.

The new recipe records the original answer, up to three corrections using **public** execution feedback, and the successful correction conversation. Only the final correct assistant response receives loss; previous mistakes and test output are context. Hidden grading feedback never becomes a repair prompt. Independent sampled successes remain complete-code examples, without being mislabeled as successful corrections.

Each repair prompt contains the original task, the latest failed program and its public-test feedback. Earlier full programs stay in the saved attempts instead of accumulating in the model context. This bounds history growth without cutting source code or the task description.

## REPL lessons and recursive evaluation

For a small rotating sample of verified training programs, Studio executes four steps in the existing isolated compiler container: inspect the task, reproduce the visible failure, apply the verified repair and retest, then return the patch. The starting point is the student's failing answer when available. A lesson without a reproducible public failure is excluded. These are **executed demonstrations**, not a claim that the model independently discovered those actions.

Each step becomes a separate training row with unique identity and final-response-only loss. Examples too long for the 4,096-token context are excluded without truncating their code. The dataset combines complete code, successful correction conversations, executed REPL demonstrations and optional locally checked reference functions. Each language contributes at most twice the size of the smallest language pool. Stored replay is retained; the cap applies to the training mix.

REPL targets must also fit the runner's 1,024-token action limit, including the answer terminator. Oversized targets are excluded intact. Lessons use the same pinned upstream initial messages, recursive instructions and sandbox feedback formatting as inference. These demonstrations teach sequential inspection and repair; they contain no recursive delegation targets.

Before and after training, a separate pilot runs the model's actual recursive REPL agent on ten reserved program tasks across five languages. Both models receive the same tasks and budgets: eight calls, 6,144 output tokens, 240 seconds and maximum recursion depth two. Root and child calls share the budget. The model sees only the task, editable source and public tests. Final patches receive independent grading.

An update needs a new pass in direct validation or the REPL pilot, without losing any prior pass in either. It must also preserve every prior pass in the complete matched 790-task function audit. Failed or incomplete comparisons cannot promote a candidate. Results show direct validation, REPL and function audit separately. Existing sessions retain their original recipe; create a new session for execution learning.

## Research behind the choice

- [RISE: Recursive Introspection](https://arxiv.org/abs/2407.18219) studies iterative training that uses a model's unsuccessful attempts as the context for learning a correction. Its reported tasks and models differ from this local coding experiment.
- [SCoRe: Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917) identifies distribution mismatch and behavior collapse in offline correction training. It supports collecting current-model mistakes and measuring actual multi-turn behavior. This implementation is supervised adaptation, **not SCoRe or reinforcement learning**.
- [Recursive Language Models](https://arxiv.org/abs/2512.24601) motivates persistent REPL interaction and bounded model subcalls. Greater recursion depth is not itself a training signal or proof of improved code.
- [Hugging Face SFT Trainer documentation](https://huggingface.co/docs/trl/en/sft_trainer) describes completion-only and assistant-only loss. This repository's existing tokenizer masks the prompt and all earlier turns, training only the selected final response.

These sources motivate an experiment. They do not establish a quality improvement for this model. The small REPL pilot and repeated public function tests are development checks; a fresh held-out comparison is still needed for a broader claim.

## Measured outcome on 2026-10-19

The first two completed rounds of the execution-repair experiment produced **no accepted update**. The retained model remained the base model.

| Round | Direct validation, before → after | REPL, before → after | Full function audit, before → after |
| --- | --- | --- | --- |
| 1 | 5/40 → 6/40 | 0/10 → 0/10 | 493/790 → 488/790 |
| 2 | 3/40 → 2/40 | 0/10 → 0/10 | Skipped after validation rejection |

Round 1 gained 14 function passes and lost 19. Validation includes rotating problems, so scores from different rounds are not a matched comparison. The second round trained on 436 complete-code examples, four correction conversations and 40 REPL rows. Those REPL rows represent ten scripted trajectories. Two action targets exceeded the runtime's per-action output allowance.

Across the 40 completed REPL attempts, 33 contained no recognized executable block; none made a subcall. Unrecognized `python` fences previously received no feedback, so the model often repeated them until its budget expired. A separate two-task diagnostic on existing synthetic **training** fixtures improved command execution with an explicit inspection example, but still repaired 0/2 tasks. This is evidence of a protocol problem and limited agent capability, not evidence of successful learning. The continuous experiment was paused with its data and reports preserved.

## Operational behavior

### Focused parent/child curriculum

`agent_curriculum.py` builds a separate, finite experiment from twenty authored training families. Each family has sequential and recursive variants, with and without public test files. The policy executes in the real isolated REPL; a recursive child returns a patch which the parent applies and checks. Both parent and child actions become final-response training targets. A deliberately invalid fence is omitted as a target, while its actual feedback remains in the recovery conversation. Reference solutions and private grading checks are never inserted into the model's input.

The first dataset contains 400 verified actions from 80 trajectories and 150 previously verified coding anchors. All 550 rows fit without truncation or exclusions. The largest action is 240 tokens; the largest complete trajectory uses 543 output tokens, including the deliberately invalid response. These are scripted reference demonstrations, not model-discovered solutions or reinforcement learning. The curriculum has Python agent actions; the anchors cover Python, Go, TypeScript, Rust and Dart.

The pilot starts a fresh adapter from the pinned base model. Collection and token-budget verification completed before the finite experiment was registered in Studio, so its first displayed stage is training. It does not resume an old optimizer or reuse the old REPL lessons. Any subsequent continuous session must start with preparation and collection under its own bound recipe.

`agent_pilot.py` freezes eight disjoint development repairs, fifteen separate coding-retention tasks and five confirmation repairs before training. Each model uses the same source hashes, task order, prompts and budgets. A recursive pass requires an executed child REPL returning the final patch, matching parent workspace contents and independent functional success. Text-only subcalls and unused proposals do not count. Captured child infrastructure failures invalidate the attempt instead of becoming model failures.

The development gate requires two distinct newly repaired families, at least one newly successful recursive repair, and no lost functional, recursive or retention passes. The confirmation gate requires a new held-out repair with no lost passes. At most two finite recipes are declared in the frozen configuration. A small pilot never authorizes the full loop by itself; the retained general model still requires its complete matched audit. The original experiment and its checkpoints remain separate.

The [RLM paper's training appendix](https://arxiv.org/html/2512.24601v3#A1) uses filtered teacher trajectories and documents how template errors harmed distillation. This local curriculum tests a smaller, authored action dataset on different coding tasks; its results must be measured independently.

### Reply termination correction

The first candidate exposed a separate inference defect: the pinned Qwen3.5 configuration stopped at document token 248044, while its [chat tokenizer](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer_config.json) ends assistant turns with token 248046. The adapter could emit a valid action and then continue into invented conversation turns. The server now uses both termination tokens for generation and response accounting.

The affected comparison was paused and preserved. The corrected comparison reran both the base and the completed adapter with identical settings; an improvement from fixing inference is not counted as a training gain. Earlier benchmark baselines cannot authorize promotion under these changed generation conditions. The completed training data and adapter remain usable because the training targets already contained the correct turn terminator.

### Completed parent/child pilot on 2026-10-19

Both predeclared recipes finished 200 training steps from the base model with all 550 rows included and no exclusions. The second recipe reused the exact frozen development suite and complete corrected baseline; it did not continue the first adapter or its optimizer. Only the declared learning rate changed.

| Model | Learning rate | Repairs | Genuine recursive repairs | Coding-retention subset |
| --- | --- | --- | --- | --- |
| Base | — | 0/8 | 0/4 | 0/15 |
| Candidate 1 | 0.00005 | 0/8 | 0/4 | 0/15 |
| Candidate 2 | 0.0001 | 0/8 | 0/4 | 0/15 |

The four recursive repairs are a subset of the eight repairs. Candidate 1 returned two patches, both of which failed independent tests; five other attempts exhausted the action budget and one reached the consecutive-error limit. Candidate 2 returned one failing patch and exhausted the action budget on its seven other attempts. Independent verification checked the complete task order, source/model/data bindings, runtime binding and child-patch evidence, and reran every returned patch through the isolated grader.

**No candidate qualified, no adapter was promoted, and the full loop was not restarted.** The five held-out confirmation repairs and fresh 790-task audit were not run because the development gate failed. The 15-task retention subset also had no baseline passes, so it cannot establish preserved general coding ability. This is a negative result for these two small supervised recipes, not proof that recursive training cannot work.

The [public result summary](../reports/agent-curriculum-pilot-20261019.json) contains scores, settings, failure counts and evidence hashes. Private examples, traces and model weights remain local. Both completed experiments and checkpoints are preserved; earlier continuous sessions remain paused and the GPU lock is released. Studio shows the completed verdict and REPL result instead of a pending-comparison message, and lists the finite pilot's saved checkpoints.

All runtime data, prompts, traces and model weights stay under the ignored `.cache` directory. The original worker owns the GPU lock and performs cleanup on Pause. The REPL uses the prepared `.cache/rlm-env` environment; model serving and adapter training use `.cache/qwen35-env`. New session preparation requires both environments. No dependency or model download occurs during the offline worker run.

Invalid coding-stage requests stop with the server's error reason and require attention. Temporary server errors, request timeouts and rate limits remain retryable. A coding stage uses exit code 2 for invalid input so the worker does not repeatedly reload the model for a request that cannot succeed.

An unrecognized REPL response now receives explicit format feedback without executing it. The upstream agent's time and cell-error limits count as failed model attempts; they do not trigger worker retries. Existing experiments retain their frozen source and comparisons; these changes require a separate experiment.

Home shows first-attempt passes and verified repairs from the current collection, the number of tokenized REPL training rows, current task and accepted updates. The decision card explains lost passes by suite. A training-row count is not a count of independent tasks or evidence of better quality.
