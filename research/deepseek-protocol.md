# DeepSeek source review and explicit tool actions

The completed short-action adapter comparison lost five functional passes and two
tested ordinary REPL passes, with no recursive gain. The separate direct-JSON
comparison also lost one functional pass. Those results remain unchanged.

The next bounded experiment tests an inference interface on retained BASE only.
It does not train an adapter, use the eight reserved learning heldouts, or
authorize a full Collect restart. Model results are not available yet.

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

The finite controller runs comparison followed by CPU context replay and
independent patch regrading. Manual inspection of meaningful executed assertions
is still required. Full results, individual losses and limits must be published
before any conclusion. The prior learning gates remain in force.
