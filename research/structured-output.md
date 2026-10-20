# Optional structured output

The Qwen3.5 BF16 evaluation server can constrain JSON generation with XGrammar.
It also validates the completed answer against the requested schema. The capability is
off by default and does not alter ordinary text requests, REPL prompts, training targets,
or existing frozen experiments.

This adopts two useful principles from
[DeepInfer](https://github.com/deepseek-ai/DeepInfer) and
[deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe):
enforce a requested format during decoding, and distinguish protocol validation from
model execution. DeepInfer itself uses
[XGrammar](https://github.com/deepseek-ai/DeepInfer/blob/main/requirements.txt).
Our integration uses XGrammar's existing Transformers adapter with the Qwen tokenizer,
full model vocabulary and both model/chat stop tokens. No DeepSeek model runtime or
DeepSeek-specific prompt encoding is imported. No source code from either DeepSeek
repository is vendored.

## Use

In a separate prepared Qwen evaluation environment, install
`requirements-structured.txt` and start `qwen35_server.py` with its usual pinned revision
and `--structured-output`. The health endpoint reports `structured_outputs`.
The server stays on loopback. Missing dependencies fail startup explicitly.

Both `/v1/chat/completions` and `/batch` accept an optional `response_format`.
An absent format or `{"type":"text"}` keeps normal generation.
`{"type":"json_object"}` requests an object. A strict schema request is:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "patch",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "solution.py": {"type": "string"}
        },
        "required": ["solution.py"],
        "additionalProperties": false
      }
    }
  }
}
```

The supported schema subset has an object root, nested objects/arrays, primitive types,
enums, descriptions, properties, required keys, boolean `additionalProperties`, and
array `items`. It is bounded to 16 KiB and 16 nesting levels. References, regexes,
formats, composition and other keywords are rejected explicitly. Schemas do not fetch
external resources. Schema semantics are preserved when additional properties are unspecified.
This is a deliberately small contract, not full OpenAI protocol compatibility.

Each generation batch gets fresh grammar matchers; only compiled grammars are cached,
up to 16 entries. A request never falls back silently to unconstrained output.
The response includes:

```json
{"structured_output": {"valid": true, "error": null}}
```

Consumers must require `valid: true` before treating the answer as schema-compliant.
A token-limited answer is `incomplete`, even if its current text parses. Other reasons
are `invalid_json` and `schema_mismatch`. Duplicate keys, non-finite numbers and numbers
that overflow a finite float are rejected. Raw text, token usage and finish reason remain
available; no repair or retry is hidden in these results.

Valid JSON can contain incorrect Python. This feature supplies no solution, tests,
delegation or semantic verification. Existing path checks, syntax checks and functional
grading remain necessary.

## Independent comparison

`python -m research.structured_compare` runs four public format canaries, including two
multi-file repairs without supplied tests. These are separate from the eight reserved
learning-confirmation tasks. It reuses one retained BF16 base model, with no adapter,
and identical prompts, sampling, token budgets and batch size. Only the constraint
changes. Order alternates between tasks. The measured time includes compilation and
generation overhead; these four pairs are not a throughput benchmark.

The runner requires `--revision`, a new `--output` directory, the installation's shared
`--gpu-lock`, and `--after-session` pointing to a completed experiment with a stopped
worker. Run it only after any subsequent confirmation/audit has also released the GPU.
It refuses existing output, competing GPU locks and existing model servers.

Reports bind source hashes, model revision and package versions. Each response is
independently checked for complete valid JSON, Python syntax and isolated Docker
functional success. Partial or duplicated pairs cannot become a summary. Functional
gains and losses are listed individually. Both `learning_gain_established` and
`full_loop_authorized` remain false regardless of the result.

The integration passed 65 targeted checks in the separate Linux environment, including
actual CPU grammar masks, multiple stop tokens, Unicode/escaping, padded vocabulary,
fresh matchers, bounded requests, truncation and existing server/experiment regressions.
A separate CPU round trip with the pinned Qwen3.5 tokenizer passed. These checks do not
constitute a model before/after result.

The separate GPU comparison is complete. It ran once after the learning candidate
failed its development gate and released the GPU. All eight responses were independently
checked against the schema, compiled and regraded in the isolated Docker environment,
including the two responses containing invalid Python. Source/runtime bindings, task
order, identical prompt hashes and token counts matched the frozen comparison.

| Measure | Plain | Schema constrained |
| --- | ---: | ---: |
| Complete valid JSON matching the schema | 4/4 | 4/4 |
| Valid Python syntax | 3/4 | 3/4 |
| Functional repairs | 2/4 | 1/4 |

There were no functional gains and one loss, on the public normalized-records task.
The constrained answer joined words with `" - "` instead of `"-"`, violating the
requirement. The fixed-literal task returned the same invalid Python in both modes,
even though its enclosing JSON was valid. All responses ended normally within the
output budget; no truncated answer was counted as success.

This small deterministic comparison provides no evidence of a format benefit on
these tasks and records a functional regression. Constraints remain opt-in and
disabled by default. Four public direct-patch tasks do not establish general model
quality, REPL ability, recursive learning or throughput. The eight reserved learning
tasks remain unused, and the full training loop remains paused.
See the [complete aggregate results](../reports/structured-output-20261020.json).
