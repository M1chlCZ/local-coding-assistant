# Windows experiment recipe

This recipe builds the pruned Qwen3.6 derivatives.
The launcher default uses upstream Qwen3.8-27B.
See the [RLM experiment](rlm.md) for recursive coding and the finite search over agent settings.
See the [adapter recipe](adapter-training.md) for the local training path.
The [Windows learning session](learning-session.md) adds tested repair examples, short training candidates, and quality-based stopping.

Run PowerShell from the project folder on the PC.
Use Python 3.14, Git, and a CUDA GPU for research.
Allow several hundred GB of free disk space.
The pinned source model occupies 71.92 GB.

## 1. Prepare the environment

```powershell
py -3 -m venv .cache/research-env
$python = "$PWD/.cache/research-env/Scripts/python.exe"
& $python -m pip install torch==2.14.0+cu130 --index-url https://download.pytorch.org/whl/cu130
& $python -m pip install accelerate==1.14.0 compressed-tensors==0.17.1 datasets==5.0.0 huggingface-hub==1.23.0 numpy==2.5.1 safetensors==0.8.0 tqdm==4.68.4 transformers==5.13.1 pytest==9.1.1 sentencepiece==0.2.2 protobuf==7.36.2
git clone https://github.com/anik-jha/moep.git .cache/moep
git -C .cache/moep checkout fdc9a3fc025236bfbc48e050e23e46d0562a5956
git -C .cache/moep apply ../../research/moep-windows.patch
Copy-Item research/test_offload.py .cache/moep/tests/test_offload.py
& $python -m pip install --no-deps -e .cache/moep
git clone --branch b11146 --depth 1 https://github.com/ggml-org/llama.cpp.git .cache/llama.cpp
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '12'
```

The patch corrects Windows support and captures expert scores during disk offload.
The regression compares resident weights with CPU and CUDA disk offload.

Run the upstream tests:

```powershell
Push-Location .cache/moep
& $python -m pytest -q tests/test_offload.py tests/test_profiler.py tests/test_surgery.py tests/test_scoring.py tests/test_gemma4.py
Pop-Location
```

The recorded result was 26 passes and two skips.
These synthetic tests cover compression mechanics, not model quality.

## 2. Download and profile

```powershell
py -3 research/download_source.py
py -3 research/calibration.py
$env:HF_HUB_OFFLINE = '1'
& $python research/profile_pilot.py
```

The calibration contains 64 coding, 32 tool-use, and 32 general examples.
Snapshot hashes detect changed data.
Dataset Viewer requests do not guarantee pinned revisions.

Profiling uses 31,232 tokens in 122 sequences.
Its weight budgets are 10 GiB on the GPU and 2 GiB in RAM.
Disk offload stores the remaining weights.
The profiler rejects nonfinite scores.

## 3. Select and prune experts

REAP ranks routing-weighted expert output norms.
It does not isolate exclusively coding experts.
The two candidates retain 192 or 128 experts per layer.
Shared experts and attention tensors remain unchanged.

```powershell
& $python -m moep.cli score --run .cache/research/calibrated --criterion reap --keep-ratio 0.75
& $python -m moep.cli prune --run .cache/research/calibrated --model .cache/research/base --selection .cache/research/calibrated/keep_reap_0.75.json --out .cache/research/reap75
& $python -m moep.cli score --run .cache/research/calibrated --criterion reap --keep-ratio 0.5
& $python -m moep.cli prune --run .cache/research/calibrated --model .cache/research/base --selection .cache/research/calibrated/keep_reap_0.5.json --out .cache/research/reap50
```

The surgery compares surviving router rows and expert tensors with the source.

## 4. Convert and quantize

Prepare the runtime and output folders:

```powershell
py -3 launcher.py setup --runtime-only
New-Item -ItemType Directory -Force .cache/research/gguf, .cache/models | Out-Null
```

Use the same converter and Q4_K_M quantizer for all candidates:

```powershell
& $python .cache/llama.cpp/convert_hf_to_gguf.py .cache/research/base --outfile .cache/research/gguf/intact-bf16.gguf --outtype bf16 --no-mtp --use-temp-file
& .cache/runtime/b11146/llama-quantize.exe .cache/research/gguf/intact-bf16.gguf .cache/models/Qwen3.6-intact-Q4_K_M.gguf Q4_K_M 4
& $python .cache/llama.cpp/convert_hf_to_gguf.py .cache/research/reap75 --outfile .cache/research/gguf/reap75-bf16.gguf --outtype bf16 --no-mtp --use-temp-file
& .cache/runtime/b11146/llama-quantize.exe .cache/research/gguf/reap75-bf16.gguf .cache/models/Qwen3.6-local-code-reap75-Q4_K_M.gguf Q4_K_M 4
& $python .cache/llama.cpp/convert_hf_to_gguf.py .cache/research/reap50 --outfile .cache/research/gguf/reap50-bf16.gguf --outtype bf16 --no-mtp --use-temp-file
& .cache/runtime/b11146/llama-quantize.exe .cache/research/gguf/reap50-bf16.gguf .cache/models/Qwen3.6-local-code-reap50-Q4_K_M.gguf Q4_K_M 4
```

Conversion omits MTP and the visual projector.
This experiment uses no importance matrix or recovery training.

After quantization succeeds, register the files:

```powershell
& $python research/register_model.py Qwen3.6-intact-Q4_K_M.gguf 'Intact Qwen3.6 comparison' 1.0 intact
& $python research/register_model.py Qwen3.6-local-code-reap75-Q4_K_M.gguf 'Local coding pilot reap75 Q4_K_M' 0.75 reap75
& $python research/register_model.py Qwen3.6-local-code-reap50-Q4_K_M.gguf 'Local coding pilot reap50 Q4_K_M' 0.5 reap50
```

Registration records file sizes, SHA256 hashes, and memory settings.
Copy `research/manifest-reap50.json` to `manifest.json` for the smaller default model.
Download URLs remain null until the weights are published.

## 5. Compare models

Run each model with `launcher.py --manifest <path> serve`.
Use the same sampler settings for each comparison.
Docker is required for evaluation.

Run the reference tests and fixed HumanEval subset:

```powershell
py -3 benchmark.py --verify-only --output reports/humaneval-reference.json
py -3 benchmark.py --base http://127.0.0.1:8080 --label candidate --output reports/humaneval-candidate.json
```

Use `evaluation.py` for coding tasks and repository repair.
Use `context_check.py` for 8K, 16K, and 32K retrieval trials.
Both accept `--base http://127.0.0.1:8080`.
Record memory use and failures.

The original comparison used repetition penalty 1.0.
The final launcher uses 1.1 over 256 tokens.
Keep those results separate.

See [MODEL_CARD.md](../MODEL_CARD.md) for results and limits.
Preserve the [license notices](../NOTICE.md) with model releases.
The calibration sources require a separate license review.
