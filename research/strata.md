# Strata teacher pilot

[Strata](https://github.com/Niko1221/Strata) supplies local inference and an API. It does not train the student.
This pilot used its Coder IQ1_M model through CUDA on Windows, with CPU and RAM offloading.
The source revision was `d9ab8435f654c368c586340d490915f6addf56a3`. The engine version was `0.1.35`.
Setup used an 8,192-token context, resident low-RAM mode, and no vision.
The server listened on loopback with a private API key.

## Measured result

Hardware: RTX 5070 Ti with 16 GB VRAM, Ryzen 9 9900X, and 32 GB RAM.
The [aggregate report](../reports/strata-2026-10-02/summary.json) records the source hashes and limits.

| Teacher | Successful repairs | Total task time |
| --- | ---: | ---: |
| Existing Qwen3.8-27B, UD-Q4_K_M | 11/15 | 720.38 seconds |
| Strata Coder, IQ1_M | 3/15 | 164.77 seconds |

Both used the same 15 development tasks, REPL prompt, and per-task limits.
Each task had eight model calls, 4,096 output tokens, and 120 seconds.
Thinking was disabled. The original holdout stayed unused.
Twelve Strata attempts exhausted their call or token budget. Several replies supplied text without executable REPL blocks.
An additional API probe returned a native `repl` tool call with the code in its arguments.
The pinned RLM client reads only `message.content`, so it discards that tool call.
This explains an interface failure in the probe. This pilot did not benchmark tool-call conversion.
These results measure this interface. They do not rank general coding ability or compatibility with other agent tools.
Neither result demonstrates recursive delegation.

Strata reached 15,868 MiB of VRAM and 30.72 GiB of total host RAM use.
Available host RAM briefly reached 0.19 GiB. This leaves little room for other applications on a 32 GB PC.
The decode log had a median of 99.5 tokens per second across 115 responses, including short repeated failures.
That number does not measure successful repair speed.
The configured context was 8K. This pilot did not measure long-context quality.

The public-data pilot therefore keeps Qwen3.8-27B as its teacher.
Strata remains an optional local experiment. It does not replace the default chat server or the student adapter.

## Install separately

Stop training and model servers first.
Use the upstream [setup instructions](https://github.com/Niko1221/Strata/blob/d9ab8435f654c368c586340d490915f6addf56a3/docs/AI_SETUP.md).
For the settings in this pilot, run these commands from the Windows project folder:

```powershell
git clone https://github.com/Niko1221/Strata.git .cache/strata-runtime
git -C .cache/strata-runtime checkout d9ab8435f654c368c586340d490915f6addf56a3
Push-Location .cache/strata-runtime
cmd /c START-HERE.bat --yes --family coder --model IQ1_M --context 8192 --vision no --low-ram resident --host 127.0.0.1 --port 8091 --no-start
Pop-Location
```

Setup downloads large model files. Use the same command after an interrupted download.
Keep the engine, runtime, and referenced model files. The native pack does not replace every GGUF file.
If a bridge or tunnel provides access, keep an API key in the private settings.
Strata code uses MIT. Its model files have separate licenses.
This repository redistributes no Strata weights.
