# Local coding derivatives of Qwen3.6

Two experimental text-only derivatives of `Qwen/Qwen3.6-35B-A3B`, built and tested locally. They are not newly trained foundation models. Base weights are Apache-2.0; our original launcher/evaluation code is MIT.

| Artifact | Routed experts kept in each layer | Quantized size | SHA256 |
| --- | ---: | ---: | --- |
| Qwen3.6-local-code-reap75-Q4_K_M.gguf | 192 of 256 | 16,269,907,840 bytes | `5f8300a7388240942f1cc26b2097e24274ed189bff28ea2bc9d739bc911652b3` |
| Qwen3.6-local-code-reap50-Q4_K_M.gguf | 128 of 256 | 11,373,057,920 bytes | `eb126d1f95439b6aaa6a0c09c48cff331cf6b1e2e67492cdbba21ba3b6fa0d9a` |

Base revision: `995ad96eacd98c81ed38be0c5b274b04031597b0`. Both retain eight active routed experts per token. Shared experts and attention tensors are copied unchanged. MTP is omitted. Text GGUF conversion omits the visual projector.

Compression reproduces/adapts REAP expert selection from existing moep research, pinned commit `fdc9a3fc025236bfbc48e050e23e46d0562a5956`. Selection uses mean routing-weighted expert output norms. It does not extract a separable set of exclusively coding experts. Windows/Accelerate offload fixes and their regression checks are included.

Calibration: 128 mixed coding, function-calling and general instruction rows, with deterministic windows yielding 31,232 tokens. Sources, terms, snapshot hashes and exact window/selection hashes are recorded under `research/`. Calibration data is excluded from this bundle. This small pilot does not reproduce the paper's full calibration.

Each checkpoint passed 360 sampled retained-row/slab integrity checks. Upstream synthetic and offload regressions passed 26 checks, with two real-checkpoint tests skipped. These validate implementation mechanics, not coding quality.

Conversion and quantization: llama.cpp b11146, commit `7fe450e19305b828c199d602c23a8337aaa1f03b`; BF16 text export followed by Q4_K_M. No importance matrix, fine-tuning, distillation or recovery training. Full source/download hashes and reproduction commands are provided.

Intended use: local English coding chat and tool use through an external harness. English is requested through prompting; the model remains multilingual. Tested hardware: RTX 5070 Ti 16 GB, Ryzen 9 9900X, 32 GB RAM, Windows 11. CUDA is active. The smaller derivative loads all 41 model layers onto GPU with CPU expert offload disabled; the optional larger candidate uses CPU expert offload in eight of 40 blocks. The smaller model is the faster research preset, using repetition penalty 1.1 over 256 tokens. The launcher now defaults to downloadable Qwen3.8-27B after the recent-model comparison in research/candidates.md.

Quality evidence and failures are in `reports/` and summarized in the README. Fixed 32-task HumanEval results: intact 31/32; 25%-pruned 30/32; 50%-pruned 29/32. Practical pilot: 5/5, 5/5 and 4/5 respectively. Both pruned models exhibited severe repetition on one parser task at some memory settings; both derivatives exhausted the output budget on one benchmark task. The larger model scored 4/5 in the practical pilot with eight CPU expert layers, versus 5/5 with four. Standard repetition penalty 1.1/256 was investigated using this known failure: it stopped the loop; the larger model still missed an empty-string edge case, while the smaller model passed the practical pilot. With this sampler, the smaller and larger models scored 29/32 and 30/32 respectively, with practical scores of 5/5 and 4/5. Tuning reports are separate from the original matched-quantization comparison. One successful bounded repair does not establish general autonomous coding reliability.

Synthetic 8K/16K/32K retrieval tests are capacity checks, not long-context coding benchmarks. Small, single-run scores cannot establish statistical equivalence to the intact model. Original public HumanEval tasks may have appeared in pretraining.

Release status: weights are built on the test PC and hash-verified; no public derivative download or GitHub release exists yet. Derivative manifest URLs remain null until their artifacts are published. The launcher default downloads an upstream Qwen3.8 GGUF, not either derivative described here. The code repository is public at https://github.com/M1chlCZ/local-coding-assistant.

For distribution, include the original Apache license (`research/qwen.LICENSE`) and the modification/attribution notice (`NOTICE.md`) alongside each GGUF. The base license permits derivatives subject to its conditions. Calibration-source terms require separate attention; this card does not certify legal clearance of every underlying dataset item.
