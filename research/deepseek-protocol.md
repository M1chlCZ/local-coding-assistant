# DeepSeek source review and explicit tool actions

The completed short-action adapter comparison lost five functional passes and two
tested ordinary REPL passes, with no recursive gain. The separate direct-JSON
comparison also lost one functional pass. Those results remain unchanged.

The completed bounded experiment tested an inference interface on retained BASE only.
It does not train an adapter, use the eight reserved learning heldouts, or
authorize a full Collect restart. Neither new interface improved task solving.
The [aggregate report](../reports/deepseek-protocol-20261020.json) includes all
18 trajectories, individual losses and the verification incident described below.

## What the source review found

The review used DeepInfer commit
`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` and deepseek-recipe commit
`935817c66f5d6e64838f419692545f06787e485f`.

| Principle | Evidence and local use |
| --- | --- |
| Treat tool calls as a distinct protocol | Recipe selects a tool-call constraint separately from a final-response constraint. Our research driver dispatches explicit `show`, `read`, `write`, `check`, `delegate`, `apply`, and `finish` actions. |
| Preserve incomplete-output status | Recipe retains a length stop even inside a partially parsed tool call. Our driver rejects a truncated action, including a syntactically complete JSON object with a length stop. |
| Separate grammar compilation from mutable decoding state | DeepInfer compiles grammars and creates request-specific matchers. We retain the existing bounded compiler cache and fresh matcher per generation. |
| Preserve action-before-argument order | We found that our cache key serialization sorted schema properties. XGrammar then required `path`, `text`, `tool`, forcing argument generation before the action name. The port now preserves declared property order. A real CPU token-mask regression test reproduced and verified this change. |
| Encode model control tokens separately from ordinary text | Recipe tracks explicit special-token ranges and encodes ordinary text without the added vocabulary. This is useful future hardening, but is **not implemented** here: a correct Qwen port must preserve its own training/template conventions. |
| Measure reproducibility rather than assume it | DeepInfer tests batched and single-request equivalence and has sequence-specific sampling seeds. This experiment holds one BASE model instance, greedy decoding, batch size one and a rotating interface order. It does not claim batch-invariant Qwen kernels. |

Source details: Recipe's [constraint selection](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/protocol/output_constraint.rs),
[stream finish handling and truncation tests](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/stream/processor.rs),
[token encoding](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe-encoding/src/tokenizer.rs);
DeepInfer's [matcher construction](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/structure_output/so_matcher_builder.py),
[grammar compiler](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/structure_output/compiler.py),
and [batch-invariance test](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/tests/models/test_batch_invariant.py).

DeepInfer's SWA bounded replay reconstructs architecture-specific attention state;
it is not a generic way to discard arbitrary conversation history. Its published
runtime targets DeepSeek-V4.1-Flash on SM100 or Ascend 950. We retain Qwen3.5-4B,
its native tokenizer/template, and the existing BF16 runtime on SM120.
See the [deployment explanation](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md)
and [supported setup](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md).

These are independently implemented adaptations of the cited principles, not a
DeepSeek model template or server transplant. DeepInfer is Apache-2.0 and Recipe
is MIT; no DeepSeek source code is vendored.

## Local failure addressed

The previous agent could create a correct child patch while its actual test call
failed. At the iteration limit an upstream fallback then requested a final JSON
answer. That answer was neither an observed successful `finish()` nor a tested
parent/child chain. Some fallback answers were fenced JSON, which the strict
parent transport rejected.

The experimental driver has no final-answer fallback. A child returns a proposal
only after the worker reports a completed `finish()`. Otherwise the parent gets
an error. The host serializes dispatch arguments; code remains data until it
reaches the existing restricted Docker REPL. The parent still has to apply and
independently check the proposal. Invalid actions and test failures remain in the
trace. Infrastructure failures fail the experiment.

This is a small research protocol over existing helpers, not an implementation
of the OpenAI tools API or DeepSeek DSML. JSON escaping is still part of the
model's action payload. Optional grammar masks address its structure; neither
the masks nor the dispatcher supply repairs or test assertions.

## Matched interface experiment

Six newly authored public development canaries each require changes to two Python
files. Three use ordinary REPL solving and three require a genuine child and
parent test chain. No test files are provided to the model. Hidden grader code
and reference patches stay outside its projected task.

The three interfaces are current fenced REPL, explicit actions without
constraints, and the same explicit actions with schema constraints. The two
explicit-action variants receive identical initial prompts. The fenced
comparison also changes the prompt and controller, so it measures the whole
interface change rather than the isolated effect of grammar masks.

All arms use retained BASE without an adapter, 8,192 context tokens, at most
1,024 output tokens per action, 12 shared parent/child calls, 8,192 shared output
tokens, 360 seconds and at most eight iterations per environment. Source hashes,
runtime versions, raw responses, stop reasons for explicit actions, context
receipts, actual check code and source hashes are retained privately.

Before inference all six fixtures rejected the original source, accepted the
reference patch, and rejected both possible single-file omissions: 24 control
grades. Scripted Docker tests also verified a real child repair and parent
recheck, exhausted-child failure, and infrastructure-error propagation. These
are integration checks, not model gains.

## Completed results

| Measure | Fenced REPL | Typed actions | Typed + constraints |
| --- | ---: | ---: | ---: |
| Functional root repairs | 3/6 | 0/6 | 0/6 |
| Functional repairs with matching parent REPL assertions | 1/6 | 0/6 | 0/6 |
| Tested ordinary REPL | 0/3 | 0/3 | 0/3 |
| Genuine recursive completion | 0/3 | 0/3 | 0/3 |
| Model generations | 48 | 57 | 54 |
| Complete schema-valid actions | Not comparable | 40/57 | 48/54 |
| Invalid JSON, including duplicate keys | Not comparable | 16 | 0 |
| Truncated actions | Stop reason not recorded | 1 | 6 |

Both typed interfaces lost the fenced functional passes on query pairs, severity
counts and range coverage. They also lost the tested parent pass on severity.
There were no functional, tested REPL or recursive gains. The two typed interfaces
had the same final task scores. Constraints removed malformed JSON in these
trajectories, but did not prevent empty test actions, incomplete long tests,
incorrect repairs or repeated failures. Their action totals have different
denominators because execution diverged; they are not independent task samples.
All 25 complete, schema-valid Python payloads in typed mode and all 34 in
constrained mode parsed, which still did not establish functional correctness.

Manual review covered every trajectory and the executed assertions:

- All tag repairs used `lower()` where Unicode `casefold()` was required.
  Their successful self-checks omitted the distinguishing case.
- The constrained duration parser passed its own assertions but accepted input
  forbidden by the specification. The plain typed arm repeated duplicate keys.
- Both typed range children produced correctly graded repairs and ran meaningful
  assertions. Neither parent applied the proposal and completed the required
  matching test chain. No child pre-repair failing assertion was observed.
  The constrained parent also wrote a test contradicting adjacent-range merging.
- Severity and flag children repeatedly reran failing assertions without fixing
  the cause. A child that exhausted its actions returned an error.
- Two fenced functional passes came from final-answer fallback patches without
  successful matching REPL checks. The severity pass had genuine parent
  assertions, but no child; it is not a recursive pass.

These results concern this small adaptation on Qwen, not DeepSeek's models or
their native tool protocol. The source review yielded useful correctness fixes,
but this interface should not become the default based on these results.

## Verification incident and independent recovery

Generation completed all 18 rows without an infrastructure failure or context
overflow. Automatic verification then failed twice: the saved effective-message
list shared an object with the list returned to the agent. Appending the next
assistant action and tool feedback also changed that receipt. The controller was
paused before further retries. Its frozen sources and evidence remain unchanged;
its automatic verification status is **failed**, not retroactively successful.

A separate CPU audit replayed all 159 requests with the pinned tokenizer. Every
replayed prompt exactly matched the independent inference trace; token counts,
eviction decisions and output budgets matched, and every request fit within
8,192 tokens. The only receipt discrepancies were the two later messages appended
to each of the 57 typed and 54 constrained receipts. It independently regraded all
nine returned root patches and both returned child patches, with no grading
discrepancies. Initial prompts matched between the typed arms. Manual inspection
also confirmed that printed success messages and zero-assertion checks earned no
tested or recursive passes.

The public code now snapshots the receipt before returning mutable messages.
A regression test failed before the fix; all 14 context/protocol checks then
passed in a separate disposable remote checkout, with no skips. No model
generation was repeated and no frozen result was edited. Existing upstream
resource warnings remain. The supplemental review supports the reported result,
but this was not a clean automatic verification run.

Full Collect remains paused. No new training recipe, reserved learning heldout
evaluation or 790-task audit was started. The prior learned candidate remains
rejected, and the learning gates remain unchanged.
