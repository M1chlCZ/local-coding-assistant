# Local adapter training

This recipe is prepared. We did not train an adapter or measure its memory use on the 16 GB GPU.

Use original Hugging Face weights. GGUF files serve inference and cannot supply this training setup.
The initial candidate is [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), pinned below. Do not train the 27B launcher model.

Export successful training runs:

```powershell
py -3 training_data.py reports/rlm-train.json --output .cache/training/round1.jsonl
```

The exporter accepts matching local evaluator reports. It verifies the task registry, splits, and hashes.
It keeps successful root calls. It excludes subcalls, duplicate calls, baseline runs, and evaluation tasks.
Report hashes detect changes. They do not prove that grading is correct.
Inspect each accepted repair before a training run. Keep reports and datasets private until their contents receive review.

Create a separate environment on the PC:

```powershell
py -3 -m venv .cache/train-env
.cache/train-env/Scripts/python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.cache/train-env/Scripts/python.exe -m pip install transformers==5.18.0 peft==0.21.1 accelerate==1.15.0 bitsandbytes==0.50.2
```

These package pins satisfy their declared dependency ranges. Their combined CUDA training path still needs a hardware test.
Stop the inference server before training.

```powershell
.cache/train-env/Scripts/python.exe train_adapter.py --dataset .cache/training/round1.jsonl --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --output .cache/adapters/round1 --check-data
.cache/train-env/Scripts/python.exe train_adapter.py --dataset .cache/training/round1.jsonl --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --output .cache/adapters/round1
```

The run uses NF4, rank 8, one example per batch, gradient checkpointing, and 50 steps.
Only final assistant tokens receive loss. Overlong examples are excluded.
The output contains the adapter, tokenizer, and training metadata. Nothing uploads automatically.

Compare repairs on the development split before promotion. Reserve the holdout split for final evaluation.
Training loss alone does not establish improvement.

Sources: [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/installation).
