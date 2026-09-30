# Attribution

- [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B): base weights, tokenizer and template. Apache-2.0. The original license is included as `research/qwen.LICENSE`; modification notices are in `NOTICE.md`. Source revision `995ad96eacd98c81ed38be0c5b274b04031597b0` is recorded in `research/source_manifest.json`.
- [Unsloth Qwen3.6 GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF): intact baseline quantization. Apache-2.0; exact artifact provenance is in `research/manifest-unsloth.json`.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): runtime, GGUF converter and browser UI. MIT. Pinned b11146, commit `7fe450e19305b828c199d602c23a8337aaa1f03b`. Runtime archives are downloaded from the official releases and checked against SHA256.
- [moep](https://github.com/anik-jha/moep): expert profiling, selection and checkpoint surgery. MIT, commit `fdc9a3fc025236bfbc48e050e23e46d0562a5956`. Based on [Half the Experts, All the Code](https://arxiv.org/html/2607.16721v1). Any adaptations must be recorded with the recipe.
- Calibration pilot: [evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) (Apache-2.0), [Hermes function calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) (Apache-2.0), and [Tulu 3 mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) (ODC-BY; underlying sources have their own terms). Exact snapshot hashes, row indices and observed Hub revisions are in `research/calibration_manifest.json`. Calibration rows are not included in this source bundle.

This project's MIT license covers its original launcher and evaluation code. It does not replace any model, dataset, runtime or upstream research license. Preserve the corresponding licenses when distributing those artifacts.

- [Recursive Language Models](https://github.com/alexzhang13/rlm): MIT library, pinned to `d04208afbad29ca675ab13478c40ee8bebc84bfe`. Installed as a dependency, with an original bounded sandbox adapter in this repository.
- [Karpathy autoresearch](https://github.com/karpathy/autoresearch): inspiration for the finite propose, measure, and retain loop. This project searches agent instructions and depth rather than editing its training program.

- [OpenAI HumanEval](https://github.com/openai/human-eval): MIT. The original test dataset and license are included under `research/`; the fixed 32-task subset is pinned in `humaneval_manifest.json`.

The 32 selected Tulu rows all identify [FLAN v2 converted](https://huggingface.co/datasets/ai2-adapt-dev/flan_v2_converted) as their underlying source. See [Tulu source terms](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) and [FLAN](https://github.com/google-research/FLAN). Dataset collection labels do not establish clearance for every original content source.


Recent comparison manifests reference Xiaomi MiMo V2.6 Distill Qwen 9B (MIT metadata, GGUF by mradermacher) and Qwen3.8-27B (Apache-2.0, GGUF by Unsloth, pinned license copy in `research/qwen38.LICENSE`). The MiMo checkpoint is an SFT of Qwen3.5-9B and its pinned repository lacks a LICENSE file. This repository does not redistribute either model. The hashes identify the upstream downloads, not models trained by this project.
