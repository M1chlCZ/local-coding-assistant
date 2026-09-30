"""Bounded local QLoRA preparation. Requires original HF weights and a free CUDA GPU."""
import argparse
import importlib.metadata
import json
import os
import re
import tempfile
from pathlib import Path

from training_data import DEFAULT_TASKS, load_verified, manifest_path, sha256

VERSIONS = {'torch': '2.14.0', 'transformers': '5.18.0', 'peft': '0.21.1',
            'accelerate': '1.15.0', 'bitsandbytes': '0.50.2'}
CHECKPOINT_FILES = ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
                    'scheduler.pt', 'rng_state.pth', 'trainer_state.json')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name+'.',
                                    suffix='.tmp', delete=False, encoding='utf-8') as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def complete_checkpoint(path, binding):
    hashes = {name: sha256(path/name) for name in CHECKPOINT_FILES}
    atomic_json(path/'complete.json', {'binding': binding, 'files': hashes})


def latest_checkpoint(output, binding):
    paths = sorted((p for p in Path(output).glob('checkpoint-*')
                    if p.name[11:].isdigit()), key=lambda p: int(p.name[11:]), reverse=True)
    for path in paths:
        if not (path/'complete.json').is_file():
            continue
        marker = json.loads((path/'complete.json').read_text())
        if marker.get('binding') != binding:
            raise ValueError('Checkpoint belongs to different data or training settings')
        if set(marker.get('files', {})) != set(CHECKPOINT_FILES) or any(
                not (path/name).is_file() or sha256(path/name) != marker['files'][name]
                for name in CHECKPOINT_FILES):
            raise ValueError('Checkpoint files failed integrity verification')
        return path
    raise ValueError('No complete checkpoint exists; start a new output folder')


def checkpoint_binding(args):
    return {'model': args.model, 'revision': args.revision, 'dataset_sha256': sha256(args.dataset),
            'manifest_sha256': sha256(manifest_path(args.dataset)), 'tasks_sha256': sha256(args.tasks),
            'max_steps': args.max_steps, 'max_length': args.max_length, 'rank': args.rank,
            'seed': args.seed, 'versions': VERSIONS,
            'trainer_sha256': sha256(Path(__file__)),
            'warm_start_config_sha256': sha256(args.warm_start/'adapter_config.json') if args.warm_start else None,
            'warm_start_sha256': sha256(args.warm_start/'adapter_model.safetensors') if args.warm_start else None}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset', required=True, type=Path)
    result.add_argument('--tasks', type=Path, default=DEFAULT_TASKS)
    result.add_argument('--model', required=True, help='Original HF causal language model, roughly 3-4B parameters')
    result.add_argument('--revision', required=True, help='Full 40-character HF commit SHA, never main')
    result.add_argument('--output', required=True, type=Path)
    result.add_argument('--max-steps', type=int, default=50)
    result.add_argument('--max-length', type=int, default=2048)
    result.add_argument('--rank', type=int, choices=(8, 16), default=8)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--check-data', action='store_true', help='Validate local data only; no packages, GPU, or downloads')
    result.add_argument('--resume', action='store_true', help='Restore the last complete checkpoint and optimizer state')
    result.add_argument('--save-steps', type=int, default=10)
    result.add_argument('--pause-file', type=Path, help='Save and stop at the next optimizer step when this file exists')
    result.add_argument('--warm-start', type=Path, help='Start a new round from an accepted local adapter; resets optimizer')
    return result


def validate_args(args):
    if not re.fullmatch(r'[a-fA-F0-9]{40}', args.revision):
        raise ValueError('Use a pinned 40-character model revision')
    if not 1 <= args.max_steps <= 500 or not 128 <= args.max_length <= 4096:
        raise ValueError('Use 1-500 steps and 128-4096 tokens')
    if not 1 <= args.save_steps <= 500:
        raise ValueError('Checkpoint interval must be 1-500 steps')
    if args.resume and not args.output.is_dir():
        raise ValueError('Resume requires an existing output folder')
    if args.output.exists() and (not args.output.is_dir() or (any(args.output.iterdir()) and not args.resume)):
        raise ValueError('Adapter output must be absent or empty')


def encode_row(tokenizer, row, max_length):
    messages = row['messages']
    # Prefix validation prevents masking the wrong tokens when chat templates differ.
    prefix_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                           enable_thinking=False)
    if not full_text.startswith(prefix_text) or len(full_text) <= len(prefix_text):
        raise ValueError('Chat template does not preserve the generation prefix')
    # Separate tokenization prevents BPE merging prompt whitespace into the first target token.
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    tokens = prefix + tokenizer.encode(full_text[len(prefix_text):], add_special_tokens=False)
    if len(tokens) > max_length:
        return None  # No truncated prompts or partial targets enter training.
    if len(tokens) <= len(prefix):
        raise ValueError('Chat template does not preserve the generation prefix')
    return {'input_ids': tokens, 'attention_mask': [1] * len(tokens),
            'labels': [-100] * len(prefix) + tokens[len(prefix):]}


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        validate_args(args)
        rows = load_verified(args.dataset, args.tasks)
        if args.check_data:
            print(f'PASS: {len(rows)} registered training rows; no model loaded')
            return
        # Optional imports keep the exporter and --help usable without a training environment.
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainerState, TrainingArguments, set_seed
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        binding = checkpoint_binding(args)
        checkpoint = latest_checkpoint(args.output, binding) if args.resume else None
        versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        for name, expected in VERSIONS.items():
            if versions[name].split('+')[0] != expected:
                raise ValueError(f'Expected {name}=={expected}, found {versions[name]}')
        if not torch.cuda.is_available():
            raise ValueError('CUDA is required; use a separate PC training environment')
        free, total = torch.cuda.mem_get_info(0)
        if free < 10 * 1024**3:
            raise ValueError('At least 10 GiB free GPU memory is required. Stop the inference server first.')
        configuration = AutoConfig.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
        # ponytail: dense text-only models up to ~4B; add other architectures after a measured training run.
        if (configuration.model_type not in ('qwen2', 'qwen3', 'llama', 'gemma', 'gemma2', 'gemma3_text')
                or getattr(configuration, 'num_experts', 0)):
            raise ValueError('Use a supported small dense text model, not a vision/MoE model')
        width = configuration.hidden_size
        layers = configuration.num_hidden_layers
        heads = configuration.num_attention_heads
        kv_heads = getattr(configuration, 'num_key_value_heads', heads)
        head_width = getattr(configuration, 'head_dim', None) or width // heads
        # Estimate loaded parameter count before requesting model shards.
        estimate = (configuration.vocab_size * width * (1 if configuration.tie_word_embeddings else 2)
                    + layers * (2 * width * heads * head_width + 2 * width * kv_heads * head_width
                                + 3 * width * configuration.intermediate_size))
        if estimate > 4.5e9:
            raise ValueError('Initial adapter recipe is limited to dense models of roughly 4B parameters')
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
        encoded = [value for row in rows if (value := encode_row(tokenizer, row, args.max_length)) is not None]
        if not encoded:
            raise ValueError('No complete training rows fit the token limit')
        if args.warm_start:
            config = json.loads((args.warm_start/'adapter_config.json').read_text())
            if config.get('base_model_name_or_path') != args.model or config.get('r') != args.rank:
                raise ValueError('Warm start requires the same base model and adapter rank')
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output/'run.json', binding)
        set_seed(args.seed)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, trust_remote_code=False, dtype=dtype,
            device_map={'': 0}, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype))
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        if args.warm_start:
            model = PeftModel.from_pretrained(model, args.warm_start, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                target_modules='all-linear', lora_dropout=0.05, bias='none', task_type='CAUSAL_LM'))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise ValueError('Tokenizer requires an EOS or pad token')
        def collate(batch):
            length = max(len(row['input_ids']) for row in batch)
            return {key: torch.tensor([row[key] + [padding] * (length - len(row[key])) for row in batch])
                    for key, padding in [('input_ids', tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]}
        settings = TrainingArguments(output_dir=str(args.output), max_steps=args.max_steps,
            per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=2e-4,
            gradient_checkpointing=True, bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
            optim='adamw_torch', logging_steps=1, save_strategy='steps', save_steps=args.save_steps,
            save_total_limit=3, report_to='none',
            push_to_hub=False, seed=args.seed, dataloader_num_workers=0)
        class Checkpoints(TrainerCallback):
            def on_step_end(self, training_args, state, control, **kwargs):
                paused = bool(args.pause_file and args.pause_file.exists())
                if paused or state.global_step >= args.max_steps:
                    control.should_save = True
                if paused:
                    control.should_training_stop = True
                atomic_json(args.output/'progress.json', {'step': state.global_step,
                            'max_steps': args.max_steps, 'pause_requested': paused})
                return control

            def on_save(self, training_args, state, control, **kwargs):
                complete_checkpoint(args.output/f'checkpoint-{state.global_step}', binding)
                return control

        trainer = Trainer(model=model, args=settings, train_dataset=encoded, data_collator=collate,
                          callbacks=[Checkpoints()])
        saved_state = TrainerState.load_from_json(str(checkpoint/'trainer_state.json')) if checkpoint else None
        recovered_complete = bool(saved_state and saved_state.global_step >= args.max_steps)
        if recovered_complete:
            # The last step can persist before final metadata; restore it without taking an extra step.
            model.load_adapter(str(checkpoint), adapter_name='default', is_trainable=True)
            trainer.state = saved_state
        else:
            trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        model.save_pretrained(args.output, safe_serialization=True)
        tokenizer.save_pretrained(args.output)
        metadata = {'schema_version': 1, 'model': args.model, 'revision': args.revision,
                    'dataset_sha256': sha256(args.dataset), 'manifest_sha256': sha256(manifest_path(args.dataset)),
                    'tasks_sha256': sha256(args.tasks), 'rows': len(encoded), 'skipped_overlength': len(rows) - len(encoded),
                    'versions': versions, 'max_steps': args.max_steps, 'max_length': args.max_length,
                    'rank': args.rank, 'seed': args.seed, 'gpu': torch.cuda.get_device_name(0),
                    'gpu_total_bytes': total, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
                    'metrics': trainer.state.log_history, 'evaluated': False,
                    'global_step': trainer.state.global_step,
                    'status': 'completed' if trainer.state.global_step >= args.max_steps else 'paused',
                    'recovered_completed_checkpoint': recovered_complete,
                    'resumed_from': str(checkpoint) if checkpoint else None}
        atomic_json(args.output/'training.json', metadata)
        print('Adapter saved locally. Evaluate it before promotion; training loss is not coding quality.')
    except (ValueError, OSError, ImportError) as error:
        argument_parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
