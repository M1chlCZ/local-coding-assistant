# Kolibri-1 candidate

Checked on 3 October 2026. This model has not been downloaded or tested by this project.

[Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1) is Aleph Alpha's English and German reasoning model.
It supports coding and tool calls.
It has 78.1 billion total parameters and 3.46 billion active parameters per token.
The active count describes computation. It does not remove the need to store the other expert weights.

Published FP8 weights need about 78 GB.
An ideal 4-bit representation of all parameters would need about 39 GB before scales, cache, activations, or optimizer state.
This estimate comes from the total parameter count. It is not a tested quantization.
The current 16 GB training setup cannot treat Kolibri as a small 3B model.
Our trainer also does not yet support its `kolibri1` architecture.

A useful later experiment is to run Kolibri on larger hardware as a teacher.
We could collect coding repairs, check them with tests, and adapt the existing small student.
Teacher answers must pass the same data and evaluation checks as other examples.
No coding improvement is established until a matched evaluation succeeds.
Expert pruning would require a separate research experiment and new quality checks.

The model card applies Apache 2.0 to the published weights and configuration files.
It excludes artifacts outside the repository from that grant.
For a redistributed adaptation, include the license, preserve applicable attribution and NOTICE content, and identify changed files.
See the [model license](https://huggingface.co/Aleph-Alpha/Kolibri-1/blob/main/LICENSE)
and [Apache 2.0 terms](https://www.apache.org/licenses/LICENSE-2.0).
