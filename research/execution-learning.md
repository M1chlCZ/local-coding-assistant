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

The [public result summary](../reports/agent-curriculum-pilot-20261019.json) contains scores, settings, failure counts and evidence hashes. Private examples, traces and model weights remain local. Both completed experiments and checkpoints are preserved; earlier continuous sessions remain paused. Studio shows completed verdicts and REPL results instead of a pending-comparison message, and lists the finite pilot's saved checkpoints.

### Why both recipes failed, and the recovery experiment

Inspection of the complete traces identified several distinct failures. The first candidate exhausted its action budget on five tasks, hit the consecutive-error limit on one, and returned two patches that failed independent tests. The second exhausted its budget on seven tasks and returned one failing patch. The transpose patch completed a real parent/child workflow and passed the visible test, but incorrectly rejected an empty input. Protocol compliance alone was therefore insufficient.

Other traces show the model preparing selected source and then omitting it and the task from the actual child request. The child cannot repair code it never receives. Repeating the same delegation consumed the remaining calls. The old reference workflow also used seven of eight shared calls for a typical recursive repair, leaving little recovery room. Its only demonstrated recovery was an invalid code fence, which did not resemble the student's actual import, test and patch failures. These observations motivate a distribution-mismatch diagnosis; they do not establish the effect of any individual change without another comparison.

There was also a separate environment defect: the persistent REPL started in `/`, despite repository files being under `/workspace`. Relative file operations failed. The worker now starts in `/workspace`; a real-container regression check reproduces the old failure and verifies the repair. This environment fix applies equally to the new baseline and candidate and cannot count as a training gain. Fresh-process public tests avoid reusing stale Python imports after edits.

The new finite `agent_recovery.py` experiment uses **corrective imitation from student states**. On the twenty training families, it executes either zero or two live base-model root actions, then completes the task with a verified reference continuation. Student actions remain in the conversation but receive no target loss. Only successful, independently graded expert continuations become lessons. Recursive lessons use the same explicit routing instruction as evaluation, pass task and source into the child, and require the parent to apply its returned patch. Compact reference continuations take two calls for ordinary repairs and five for recursive repairs, leaving room for actual student mistakes. Complete trajectories must fit the existing runtime limits without truncation. Previously verified multilingual code anchors remain in the training mix.

This is supervised corrective imitation, not reinforcement learning or model-discovered solutions. It is informed by [RISE](https://arxiv.org/abs/2407.18219), [SCoRe](https://arxiv.org/abs/2409.12917) and the [dataset-aggregation motivation of DAgger](https://arxiv.org/abs/1011.0686). Those papers do not establish that this local recipe will work.

The new experiment freezes ten development repairs (the previous eight plus two additional development families), including five recursive tasks, before collection. It runs a fresh matched baseline, one declared training recipe and the candidate comparison. The five confirmation repairs remain unused until candidate selection. A development gain still needs confirmation and a fresh complete matched general-code audit before any continuous loop can start from preparation and Collect. Collection success and training loss are not evaluation results.

Collection finished all 80 declared trajectories. Seventy-nine complete trajectories fit the training and inference limits; one whole trajectory was excluded because its student response exceeded the action/context preflight. The resulting dataset has 278 corrective REPL targets and 150 verified coding anchors, with student-state coverage across all twenty training families. All 428 selected rows fit without truncation. Independent collection checks verified target masking, family separation, all trace hashes, and fresh execution of all twenty reference repairs and their failing controls. Forty trajectories contained actual student prefixes; twenty-nine of those prefixes contained an execution error. These checks validate the data, not model quality.

### Completed corrective-imitation comparison

The single declared recipe completed 162 steps from a fresh base adapter at learning rate 0.00005 and rank 8. Three epochs determined the effective step count below the 200-step cap. All 428 rows were included, with no truncation or selected-row exclusions.

| Model | All repairs | Ordinary REPL repairs | Genuine recursive repairs | Coding-retention subset |
| --- | --- | --- | --- | --- |
| Fresh matched base | 0/10 | 0/5 | 0/5 | 0/15 |
| Corrective-imitation candidate | 2/10 | 2/5 | 0/5 | 0/15 |

The ordinary and recursive columns partition the ten repair tasks. The candidate repaired the date-span and idle-expiry development families, each in two REPL calls. Independent verification checked the complete matched task order, frozen sources, runtime, data and adapter bindings, and regraded all three returned patches. Two passed; the third failed. This demonstrates a small ordinary REPL gain on these authored development tasks, with no demonstrated recursive gain.

All five recursive attempts exhausted the shared eight-call budget. The inspected child requests now include task and source, but the proposed algorithms remain incorrect: an ISBN proposal omits the checksum, a Roman-numeral proposal uses incorrect symbols, and an edit-distance proposal raises an index error. After public tests reject these patches, the parent repeats inspection, delegation or application until its budget expires. The other two unsuccessful ordinary attempts also exhaust the budget. Better protocol execution has not yet produced reliable recursive problem solving.

**The development gate failed because it required at least one newly successful recursive repair.** No candidate was selected or promoted, and every full learning loop remains paused. The five held-out repairs and fresh matched 790-task audit were not run after this rejection. No losses were observed in the small development comparison, but the retention subset had no baseline passes and cannot establish preserved general coding ability. The working-directory repair applied to both models and is not counted as a gain.

The [verified public result](../reports/agent-recovery-pilot-20261019.json) records scores, recipe, failure counts and evidence hashes. Raw traces, training data, checkpoints and model weights remain private and preserved. This finite recipe is complete; it does not authorize restarting the continuous loop from Collect.

### Execution-tested recursive distillation

The next investigation found a concrete gap: the recovery dataset's 80 child targets across twenty families never execute the proposed repair. The parent sees failed tests, but the child has learned to return an untested algorithm. This is a plausible contributor to failure, not an isolated causal finding.

`agent_tested.py` prepares 128 distinct training function families from the pinned, locally verified OpenCodeInstruct source. Two publisher assertions become public checks; the remaining assertions stay in independent grading. Benchmark names and text overlaps are excluded, along with reserved repair families and near copies. These filters cannot establish absence of semantic or pretraining overlap.

For each family, the collector obtains one actual answer from the recovery adapter. Answers failing both public and private checks become additional repair starts. Declared stub starts provide coverage independently of student success. In every recursive lesson, the child reproduces the failure, implements and tests the reference repair, then returns its patch; the parent applies that exact patch and retests. Stub lessons cover tasks both with and without supplied tests. For the latter, a reference action creates public checks before repair. These are executed supervised demonstrations, not student-discovered solutions or reinforcement learning.

Only complete trajectories fitting the existing action, context, time and call limits are eligible. At least 64 distinct families must retain all four ordinary/recursive and public/source-only stub variants. The mix also retains 150 verified multilingual code anchors. Actual failed student code remains visible as repair context; private grading feedback never becomes a prompt.

One recipe starts from the previous recovery adapter with a fresh optimizer: rank 8, learning rate 0.00002, two epochs capped at 250 steps, accumulation 8, context 4,096 and seed 3407. Both the fresh baseline and candidate receive identical new instructions and inference settings. The required gain is at least two newly repaired development families, including a genuine recursive repair, with no lost passes. A qualifying candidate must be selected before the five untouched confirmation tasks. It must then pass those checks and a fresh complete 790-task comparison against the retained base with no lost base passes. Only then may a new continuous session start from preparation and Collect. The old adapter remains unpromoted.

The ordinary and recursive reference workflows passed real-container checks, including child failure/success execution and exact parent application, with and without supplied tests. These checks validated lesson execution; learning required the separate matched comparison below.

Collection exposed another defect before training: one real student answer produced enough test output to exceed the sandbox's 8,192-character cell limit. The collector retried this deterministic failure instead of recording an excluded lesson. The corrected collector preserves the failed trace, exports no targets from it, and excludes the whole trajectory. Infrastructure errors still stop collection. The original frozen attempt stays paused and preserved; a separate corrected snapshot reuses the 128 saved student answers with unchanged evaluation tasks, budgets, recipe and minimum-family requirement. This is a collection repair, not a measured model improvement.

The corrected attempt's preparation guard then caught a different source selection. Rescanning the shard admitted two functions whose publisher tests vary with random draws or set iteration order; six fresh executions reproduced both passing and failing outcomes for each. The saved student answers remained bound to the original 128 functions. All 128 original reference solutions passed a fresh independent execution check, and all 128 empty-solution controls failed with missing-function errors. The mismatched rescan was archived, the original saved preparation state was restored, and the same finite trial resumed. Frozen source files, stage scripts, evaluator, recipe and student answers were unchanged. No comparison score was available at this recovery point.

### Verified development result on 2026-10-19

The completed matched comparison passed its development gate. Both models used the same new instructions, frozen tasks and inference settings. The baseline was the unpromoted recovery adapter; it was evaluated afresh. All twelve returned patches were independently regraded, and all three recursive gains included actual child execution, passing child tests, and exact parent application.

| Measure | Before | After |
| --- | ---: | ---: |
| All functional repairs | 2/10 | 6/10 |
| Ordinary REPL repairs | 2/5 | 2/5 |
| Genuine recursive repairs | 0/5 | 3/5 |
| Small coding-retention subset | 0/15 | 0/15 |

Roman numerals, header multimap and booking overlap were new genuine recursive successes. Edit distance also passed functional grading but did not meet the recursive execution criterion. There were no lost passes. Zero baseline passes in the retention subset cannot establish preservation of general coding ability.

Training completed 250 steps on 2,166 rows from 576 complete trajectories across 128 families plus 150 coding anchors. Two whole output-limit failures contributed no targets. The selected adapter was frozen before the five untouched confirmation repairs began. See the [aggregate development and confirmation evidence](../reports/agent-tested-development-20261019.json).

### Held-out confirmation rejected the candidate

The complete matched confirmation scored **0/5 before and 0/5 after**. Ordinary repairs remained 0/3 and genuine recursive repairs remained 0/2. The starting adapter returned three patches, all of which failed independent regrading; the candidate returned no final patch. Both runs completed under the frozen settings without infrastructure errors. The selected weight hash and selection-before-confirmation evidence remained intact.

These five tasks contain multiple source files and no supplied public test file. Three candidate traces assumed `test_visible.py` existed and raised `KeyError`; two also used `child_reply` before it existed. Other traces contained repeated failed assertions, malformed code or an invalid JSON reply. This shows limited transfer beyond the demonstrated workflow, rather than a generally reliable recursive solver. It does not isolate one causal training defect.

The required confirmation gain was absent, so the candidate was rejected. The full 790-task audit was not run after this failed gate, no adapter was promoted, and every full training loop remains paused. The five confirmation tasks are now used and cannot serve as untouched confirmation for further tuning. Checkpoints and private evidence are preserved; no additional recipe was launched.

[CodeRL](https://arxiv.org/abs/2207.01780) and [ReST-EM](https://arxiv.org/abs/2312.06585) provide primary research on execution-based feedback and filtering. Their methods and evaluated models differ from this supervised local experiment; they do not guarantee improvement here.

All runtime data, prompts, traces and model weights stay under the ignored `.cache` directory. The original worker owns the GPU lock and performs cleanup on Pause. The REPL uses the prepared `.cache/rlm-env` environment; model serving and adapter training use `.cache/qwen35-env`. New session preparation requires both environments. No dependency or model download occurs during the offline worker run.

Invalid coding-stage requests stop with the server's error reason and require attention. Temporary server errors, request timeouts and rate limits remain retryable. A coding stage uses exit code 2 for invalid input so the worker does not repeatedly reload the model for a request that cannot succeed.

An unrecognized REPL response now receives explicit format feedback without executing it. The upstream agent's time and cell-error limits count as failed model attempts; they do not trigger worker retries. Existing experiments retain their frozen source and comparisons; these changes require a separate experiment.

Home shows first-attempt passes and verified repairs from the current collection, the number of tokenized REPL training rows, current task and accepted updates. The decision card explains lost passes by suite. A training-row count is not a count of independent tasks or evidence of better quality.

### Multi-file tasks without supplied tests

The failed confirmation exposed a mismatch in the previous lessons: their instruction still
required `test_visible.py`, and their no-public-test variant inserted that file into context.
The new `agent_multifile.py` curriculum starts with no test files. It derives a small scratch
check from examples in the specification, executes the failure, repairs both the function
and its caller, and retests in a fresh process. Recursive lessons pass supporting import
files to the child, which executes its own checks before returning the patch. Context and
private grading checks remain unchanged. These are executed authored demonstrations, not
model-discovered solutions or reinforcement learning.

The new finite recipe starts with a fresh adapter and optimizer on the retained base model.
Training families come from the already verified train-only source snapshot; imported paths
and callers vary. A new matched baseline uses the same corrected instructions as the
candidate. Ordinary evaluation explicitly disables subcalls. Previously consumed tasks
are development material only. Eight new private, authored multi-file tasks (four ordinary,
four recursive, no supplied tests) are frozen before training and reserved until candidate
selection. Their references and both individual defect controls must pass/fail as expected.

Development and confirmation each require an ordinary REPL gain and a genuine recursive
gain without lost functional, recursive, or required retention passes. Development also
requires two distinct new repair families. Selection precedes confirmation, and the full
matched 790-task retained-base audit remains mandatory before any promotion or full-loop
restart from preparation and Collect. No success is inferred from collection, changed
instructions, reference-policy canaries, or training loss. This is a small synthetic transfer
experiment; it cannot establish production repository competence by itself.

The design follows execution-verified task and trajectory construction from
[SWE-smith](https://arxiv.org/abs/2504.21798) and the inspect/reproduce/repair/retest workflow
described in [Anthropic's SWE-bench agent report](https://www.anthropic.com/engineering/swe-bench-sonnet).
These sources motivate the experiment; their benchmark scores are not claims about this model.

The first multi-file preparation stopped before collection or model evaluation: a source
lesson's assertions accepted a no-op caller returning `None`. That snapshot is paused and
preserved. The corrected preparation explicitly checks the reference, broken caller,
specification examples, and each missing source edit. Non-discriminating lessons are
excluded with recorded reasons; infrastructure errors still stop preparation. The corrected
snapshot keeps the same recipe and the same eight unobserved confirmation tasks. This is
a data-quality correction, not a failed or successful learning comparison.

The corrected trial completed on 20 October 2026 and failed the development gate.
Collection produced 254 complete executed trajectories across 127 training families,
with 1,166 included training rows. The single recipe completed all 300 steps. An
unexpected host shutdown interrupted training; the existing worker resumed from an
integrity-checked checkpoint with its optimizer, scheduler and RNG state. Frozen
sources, settings, dataset and baseline hashes remained unchanged.

| Complete matched development check | Retained base | Candidate |
| --- | ---: | ---: |
| Functional repairs, either mode | 1/15 | 0/15 |
| Ordinary REPL repairs | 0/8 | 0/8 |
| Genuine recursive repairs | 0/7 | 0/7 |
| Small retention check | 0/15 | 0/15 |

Independent verification checked ordered task coverage, frozen bindings and checkpoint
integrity, then regraded every returned final patch in fresh containers: one baseline
patch passed and one failed, matching the saved results. The candidate returned no
final patches and lost the baseline's one functional pass. That baseline pass occurred
in the recursive condition but used no child call, so it was never counted as a genuine
recursive success.

The candidate made ten child calls and executed 32 child cells. Its two child final
answers were invalid patches. Trace inspection found malformed JSON, invalid Python,
missing context keys and repeated failed checks; some child actions used unrelated
example functions instead of the assigned repository. Eight tasks ended at the
consecutive-error limit and seven exhausted the call/token budget. These observations
show failed transfer and error recovery; they do not establish which training change
would fix it. The baseline also had one request timeout at the frozen task deadline,
which the evaluator counts as a failed bounded attempt.

The [aggregate evidence](../reports/agent-multifile-trial-20261019.json) records the
rejection and evidence hashes. No candidate was selected or promoted. The eight fresh
confirmation tasks remain unused, and the 790-task audit was not started because the
development prerequisite failed. All older loops remain paused; no full-loop restart
or additional recipe followed. Private examples, traces and checkpoints are preserved.
The all-zero small retention result provides no positive evidence of retained general
coding ability.

### Short REPL actions and recovery imitation

A new authorized trial addresses the malformed Python/JSON and repeated-error failures.
The optional action interface keeps the Python REPL but removes nested JSON construction
from source edits and child transport. The model still inspects source, writes the repair,
and supplies its own assertions. Small helpers validate editable paths and Python syntax,
run checks in fresh processes, transport the actual child result, and serialize changed
files only after a successful check on those exact files. They supply no solutions or tests.
The retained base and candidate use identical helpers, prompts and budgets; improvements
from changing the interface alone cannot count as training gains.

The new curriculum executes short actions and recovery from syntax failures, an incorrect
implementation, and a missing caller edit. Deliberately failed proposals remain in the
conversation history but are masked from training targets. Checks must reject the broken
repository and each missing edit. Complete trajectories, including masked observations,
must fit the inference and training limits. Source families remain separate from evaluation;
150 verified multilingual source-code anchors remain in the training mix. One fresh
retained-base adapter and optimizer use rank 8, learning rate 5e-5, three epochs capped at
300 steps, accumulation 8, sequence length 4096, and seed 3407.

Portable tests and six actual isolated parent/child reference canaries passed. These are
pipeline checks, not model results. The eight previously frozen multi-file confirmation
tasks have never been evaluated by either model and remain reserved for candidate selection.
Development still requires gains in ordinary REPL and genuine recursive solving, at least
two new families, and no lost passes. Confirmation must gain in both modes without losses,
followed by a fresh complete matched 790-task retained-base audit with no lost base passes.
The full loop stays paused until every requirement passes. This remains a small synthetic,
Python repository experiment; broad repository competence is unproven.

The interface choice is informed by the tool-design observations in
[Anthropic's agent report](https://www.anthropic.com/engineering/swe-bench-sonnet).
It is a hypothesis to test here, not evidence that this model has improved.
