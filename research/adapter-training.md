# Local adapter training

We completed a 50-step CUDA run on the RTX 5070 Ti through the dedicated WSL environment.
The [report](../reports/adapter-local-round1/summary.json) records memory use, training metrics, and repair results.
This small experiment verifies the training path. It does not establish a useful coding model.

The run used 18 complete examples from five successful training tasks.
Five longer rows exceeded the 2,048-token limit.
Training took 151.9 seconds. Logged loss decreased from 1.71 to 0.045.
Both the base and adapter scored 0/5 on direct repairs and 0/5 through the recursive harness.
The adapter produced five parseable recursive patches, but all failed functional checks.
Neither setting made recursive child calls.
The adapter remains a local experiment and does not replace chat.

Peak PyTorch allocation reached 7.99 GiB. Peak reservation reached 15.94 GiB.
These allocator values do not measure total physical GPU use.
The run left little GPU memory headroom on the measured PC.

Use original Hugging Face weights. GGUF files cannot supply this training setup.
The candidate is [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), pinned below.
The adapter remains separate from the 27B chat model.

For repeated collection and training, use the [Windows learning session](learning-session.md).
Its checkpoints support pause and resume within a round.

## Prepare in WSL

Open the dedicated environment as root:

```powershell
wsl.exe -d LocalCodingAssistant -u root
```

Install the compiler and Python headers for Triton:

```bash
apt-get update
apt-get install -y --no-install-recommends gcc python3-dev
exit
```

Open the environment as the project user:

```powershell
wsl.exe -d LocalCodingAssistant -u coder --cd /home/coder/local-coding-assistant
```

Create a separate training environment:

```bash
python3 -m venv .cache/train-env
.cache/train-env/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.cache/train-env/bin/python -m pip install transformers==5.18.0 peft==0.21.1 accelerate==1.15.0 bitsandbytes==0.50.2
.cache/train-env/bin/python -m pip check
```

These pins completed CUDA training on the measured PC.
Native Windows training remains unverified.

## Prepare data

Run the training tasks and export successful traces:

```bash
.cache/rlm-env/bin/python recursive_agent.py --split train --depth 2 --calls 16 --seconds 240 --output reports/adapter-local-round1/train.json
python3 training_data.py reports/adapter-local-round1/train.json --output .cache/training/local-round1.jsonl
```

For another run, use new report, dataset, and adapter paths.
The exporter verifies the task registry, splits, and hashes.
It keeps successful root calls and excludes subcalls, duplicates, baseline runs, and evaluation tasks.
Report hashes detect changes. They do not prove correct grading.
Inspect the accepted repairs before training. Successful tasks can contain incorrect intermediate reasoning.
Keep generated datasets private until their contents receive review.

## Train

Stop the chat model server before training.
Verify the dataset, then run the bounded training trial:

```bash
.cache/train-env/bin/python train_adapter.py --dataset .cache/training/local-round1.jsonl --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --output .cache/adapters/local-round1 --check-data
.cache/train-env/bin/python train_adapter.py --dataset .cache/training/local-round1.jsonl --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --output .cache/adapters/local-round1
```

The run uses NF4, rank 8, one example per batch, gradient checkpointing, and at most 50 steps.
The current trainer limits each candidate to approximately one dataset pass.
Its default learning rate is `5e-5`.
Only final assistant tokens receive loss. Overlong examples are excluded.
Prompt and continuation tokenization preserve the generation boundary.
The output contains the adapter, tokenizer, and training metadata. Nothing uploads automatically.

## Evaluate

Compare the adapter with its own unchanged base on separate development tasks.
Reserve the holdout split for final evaluation.
Training loss alone does not establish improvement.
Direct JSON repairs and recursive agent runs measure different behaviors.
Use the same harness for the base and adapter in each comparison.

The evaluation scripts use the pinned first candidate and its local adapter path.
For a direct repair comparison, run this command from the repository folder:

```bash
PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 .cache/train-env/bin/python research/evaluate_adapter_direct.py
```

For a recursive comparison, start the temporary loopback bridge:

```bash
HF_HUB_OFFLINE=1 .cache/train-env/bin/python research/adapter_eval_server.py
```

After the bridge prints `EVAL_READY`, run the controller from a second terminal:

```bash
PYTHONPATH="$PWD" .cache/rlm-env/bin/python research/evaluate_adapter_rlm.py
```

The controller compares both settings through the same bridge and Docker sandbox.
Each task receives eight model calls, 4,096 output tokens, and 120 seconds.
After evaluation, stop the bridge with Ctrl+C. Restart chat with `start-rlm.cmd`.
Existing evaluation reports cannot receive another run.

Sources: [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/installation).
