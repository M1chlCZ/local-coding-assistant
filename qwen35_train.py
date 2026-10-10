"""Finite, local BF16 LoRA training for pinned Qwen3.5-4B text-only repairs."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from collections import Counter

MODEL = 'Qwen/Qwen3.5-4B'
PACKAGES = ('torch', 'transformers', 'unsloth', 'unsloth_zoo', 'peft', 'accelerate', 'trl',
            'bitsandbytes', 'datasets', 'triton', 'xformers')
CHECKPOINT_FILES = ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
                    'scheduler.pt', 'rng_state.pth', 'trainer_state.json')
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_revision(model, revision):
    if model != MODEL or not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('Use the original Qwen/Qwen3.5-4B and its full 40-character revision')


def load_verified(dataset, manifest):
    """Validate provenance without describing publisher tests as local execution."""
    metadata = read_json(manifest)
    kind = metadata.get('verification')
    if (metadata.get('schema_version') != 1 or metadata.get('split') != 'train'
            or kind not in ('publisher_execution', 'local_execution')
            or metadata.get('dataset_sha256') != sha256(dataset)):
        raise ValueError('Training manifest, verification kind, split, or data hash is invalid')
    rows, seen = [], set()
    with Path(dataset).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'Invalid row {number}')
            task = row.get('task_id')
            if (not isinstance(task, str) or not task or task in seen
                    or row.get('split') != 'train' or row.get('language') not in ('go', 'typescript')):
                raise ValueError(f'Invalid, duplicate, or nontraining row {number}')
            messages = row.get('messages')
            if (not isinstance(messages, list) or not 2 <= len(messages) <= 64
                    or any(not isinstance(m, dict) or set(m) != {'role', 'content'}
                           or m.get('role') not in ('system', 'user', 'assistant')
                           or not isinstance(m.get('content'), str) or not m['content']
                           for m in messages) or messages[-1]['role'] != 'assistant'):
                raise ValueError(f'Invalid text messages in row {number}')
            verification = row.get('verification', {})
            provenance = row.get('provenance', {})
            if (verification.get('kind') != kind
                    or (kind == 'publisher_execution' and
                        (type(verification.get('fail_to_pass_count')) is not int
                         or verification['fail_to_pass_count'] < 1))
                    or (kind == 'local_execution' and verification.get('passed') is not True)
                    or not isinstance(provenance.get('dataset'), str)
                    or not re.fullmatch(r'[a-f0-9]{40}', str(provenance.get('revision', '')))
                    or not re.fullmatch(r'[a-f0-9]{64}', str(provenance.get('source_row_sha256', '')))):
                raise ValueError(f'Missing execution evidence or pinned provenance in row {number}')
            rows.append(row)
            seen.add(task)
    if not rows or metadata.get('rows') != len(rows) or metadata.get('languages') != dict(
            Counter(row['language'] for row in rows)):
        raise ValueError('Manifest counts do not match the training rows')
    return rows, metadata


def encode_rows(tokenizer, rows, max_length):
    """Only the final answer receives loss; oversized complete examples are reported."""
    encoded, included, excluded = [], [], []
    for row in rows:
        messages = row['messages']
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        full = tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=False, enable_thinking=False)
        if not full.startswith(prefix) or len(full) <= len(prefix):
            raise ValueError('Chat template does not preserve the answer prefix')
        prompt = tokenizer.encode(prefix, add_special_tokens=False)
        answer = tokenizer.encode(full[len(prefix):], add_special_tokens=False)
        if not answer:
            raise ValueError('A complete answer has no target tokens')
        tokens = prompt + answer
        record = {'task_id': row['task_id'], 'language': row['language'],
                  'tokens': len(tokens), 'answer_tokens': len(answer)}
        if len(tokens) > max_length:
            excluded.append({**record, 'reason': 'exceeds_context_without_truncation'})
            continue
        included.append(record)
        encoded.append({'input_ids': tokens, 'attention_mask': [1] * len(tokens),
                        'labels': [-100] * len(prompt) + answer})
    digest = hashlib.sha256()
    for row in encoded:
        digest.update(json.dumps(row, sort_keys=True, separators=(',', ':')).encode())
        digest.update(b'\n')
    report = {'rows': len(rows), 'included_count': len(included), 'excluded_count': len(excluded),
              'languages': dict(Counter(row['language'] for row in included)),
              'max_length': max_length, 'encoded_sha256': digest.hexdigest(),
              'included': included, 'excluded': excluded}
    return encoded, report


def verified_checkpoint(path, binding):
    path = Path(path)
    marker = read_json(path / 'complete.json')
    if marker.get('binding') != binding:
        raise ValueError('Checkpoint data, runtime, or recipe changed')
    if (set(marker.get('files', {})) != set(CHECKPOINT_FILES)
            or any(not (path / name).is_file() or sha256(path / name) != marker['files'][name]
                   for name in CHECKPOINT_FILES)):
        raise ValueError('Checkpoint failed integrity verification')
    return path


def complete_checkpoint(path, binding):
    path = Path(path)
    hashes = {name: sha256(path / name) for name in CHECKPOINT_FILES}
    atomic_json(path / 'complete.json', {'binding': binding, 'files': hashes})


def latest_checkpoint(output, binding):
    paths = sorted((p for p in Path(output).glob('checkpoint-*') if p.name[11:].isdigit()),
                   key=lambda p: int(p.name[11:]), reverse=True)
    for path in paths:
        if (path / 'complete.json').exists():
            # Corruption of a committed checkpoint is an error, never silent rollback.
            return verified_checkpoint(path, binding)
    return None


class Progress:
    """Liveness continues during compilation; step progress is a separate field."""
    def __init__(self, output):
        self.output = output
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.value = {'status': 'starting', 'phase': 'validate', 'step': 0,
                      'pid': os.getpid(), 'started_at': time.time()}
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)

    def update(self, **values):
        with self.lock:
            self.value.update(values, heartbeat_at=time.time())
            atomic_json(self.output / 'progress.json', self.value)

    def heartbeat(self):
        while not self.stop.wait(15):
            self.update()
            print(f"QWEN35 {self.value['status']}: {self.value['phase']}; "
                  f"step {self.value['step']}/{self.value.get('max_steps', '?')}", flush=True)

    def __enter__(self):
        self.update()
        self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        self.thread.join()
        if error is not None:
            self.update(status='failed', error=f'{kind.__name__}: {error}')


def runtime_versions():
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    for name in ('causal-conv1d', 'flash-linear-attention'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not-installed'
    return versions


def validate_tokenizer(tokenizer):
    # Transformers can construct a one-token placeholder from a partial Hub snapshot.
    if len(tokenizer.get_vocab()) < 100_000 or not tokenizer.encode('hello', add_special_tokens=False):
        raise ValueError('The pinned Qwen3.5 tokenizer cache is incomplete; finish the model download')


def load_model(model_id, revision, context):
    """Use identical native BF16 base weights and text layout for train and eval."""
    validate_revision(model_id, revision)
    from unsloth import FastLanguageModel
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('A CUDA GPU with BF16 support is required')
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id, revision=revision, use_exact_model_name=True,
        trust_remote_code=False, local_files_only=True, text_only=True,
        max_seq_length=context, dtype=torch.bfloat16, load_in_4bit=False,
        load_in_8bit=False, load_in_16bit=True, full_finetuning=False,
        fast_inference=False, device_map={'': 0})
    if getattr(model.config, 'model_type', '') != 'qwen3_5_text':
        raise ValueError('The runtime did not load the text-only Qwen3.5 architecture')
    if getattr(model, 'is_quantized', False) or getattr(model, 'is_loaded_in_4bit', False):
        raise ValueError('The experiment requires unquantized BF16 base weights')
    validate_tokenizer(tokenizer)
    return model, tokenizer, FastLanguageModel


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset', required=True, type=Path)
    result.add_argument('--manifest', required=True, type=Path)
    result.add_argument('--output', required=True, type=Path)
    result.add_argument('--model', default=MODEL)
    result.add_argument('--revision', required=True)
    result.add_argument('--max-steps', type=int, default=1000)
    result.add_argument('--max-epochs', type=float, default=1.0)
    result.add_argument('--learning-rate', type=float, default=5e-5)
    result.add_argument('--max-length', type=int, default=2048)
    result.add_argument('--rank', type=int, choices=(8, 16), default=8)
    result.add_argument('--gradient-accumulation', type=int, choices=(4, 8), default=8)
    result.add_argument('--save-steps', type=int, default=10)
    result.add_argument('--seed', type=int, default=3407)
    result.add_argument('--min-examples', type=int, default=2000)
    result.add_argument('--min-per-language', type=int, default=250)
    result.add_argument('--pause-file', type=Path)
    result.add_argument('--resume', action='store_true')
    result.add_argument('--check-data', action='store_true', help='Check manifest only; load no model')
    result.add_argument('--prepare-data', action='store_true', help='Tokenize and report exclusions; load no model')
    return result


def validate_args(args):
    validate_revision(args.model, args.revision)
    if (not 1 <= args.max_steps <= 2000 or not 128 <= args.max_length <= 8192
            or not 0 < args.max_epochs <= 3 or not 1e-6 <= args.learning_rate <= 2e-4
            or not 1 <= args.save_steps <= 100 or args.min_examples < 1
            or args.min_per_language < 1):
        raise ValueError('Invalid finite training settings')
    if args.check_data:
        return
    if args.resume and not (args.output / 'request.json').exists():
        raise ValueError('Resume requires the saved training recipe')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError('Use an empty output directory or --resume')


def make_binding(args, report, versions):
    keys = ('model', 'revision', 'max_steps', 'max_epochs', 'learning_rate', 'max_length',
            'rank', 'gradient_accumulation', 'save_steps', 'seed', 'min_examples', 'min_per_language')
    return {**{key: getattr(args, key) for key in keys}, 'precision': 'bf16',
            'dataset_sha256': sha256(args.dataset), 'manifest_sha256': sha256(args.manifest),
            'encoded_sha256': report['encoded_sha256'], 'included_count': report['included_count'],
            'target_modules': TARGET_MODULES, 'versions': versions,
            'trainer_sha256': sha256(__file__)}


def request_binding(args):
    keys = ('model', 'revision', 'max_steps', 'max_epochs', 'learning_rate', 'max_length',
            'rank', 'gradient_accumulation', 'save_steps', 'seed', 'min_examples', 'min_per_language')
    return {**{key: getattr(args, key) for key in keys},
            'dataset_sha256': sha256(args.dataset), 'manifest_sha256': sha256(args.manifest),
            'trainer_sha256': sha256(__file__)}


def train(args, rows, progress):
    # Import Unsloth first so its Transformers patches apply before model classes load.
    progress.update(status='running', phase='load_runtime')
    if not args.prepare_data:
        import unsloth
    from transformers import AutoTokenizer
    progress.update(status='running', phase='tokenize')
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                    trust_remote_code=False, local_files_only=True)
    validate_tokenizer(tokenizer)
    encoded, report = encode_rows(tokenizer, rows, args.max_length)
    atomic_json(args.output / 'tokenization.json', report)
    if (len(encoded) < args.min_examples or any(report['languages'].get(language, 0)
            < args.min_per_language for language in ('go', 'typescript'))):
        raise ValueError(f"Only {len(encoded)} complete examples fit; minimum is {args.min_examples} "
                         f"and {args.min_per_language} per language. See tokenization.json.")
    if args.prepare_data:
        progress.update(status='prepared', phase='tokenize', included=len(encoded),
                        excluded=report['excluded_count'])
        return
    import torch
    from transformers import Trainer, TrainerCallback, TrainerState, TrainingArguments, set_seed
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    versions = runtime_versions()
    binding = make_binding(args, report, versions)
    if args.resume and (args.output / 'run.json').exists() and read_json(args.output / 'run.json') != binding:
        raise ValueError('Resume data, runtime, source, or recipe changed')
    atomic_json(args.output / 'run.json', binding)
    checkpoint = latest_checkpoint(args.output, binding) if args.resume else None
    saved = TrainerState.load_from_json(str(checkpoint / 'trainer_state.json')) if checkpoint else None
    steps = min(args.max_steps, max(1, math.ceil(math.ceil(len(encoded) /
                args.gradient_accumulation) * args.max_epochs)))
    progress.update(phase='load_model', step=saved.global_step if saved else 0,
                    max_steps=steps, included=len(encoded), excluded=report['excluded_count'],
                    resumed_checkpoint=checkpoint.name if checkpoint else None)
    if args.pause_file and args.pause_file.exists():
        progress.update(status='paused', phase='before_model_load')
        return
    set_seed(args.seed)
    model, runtime_tokenizer, fast = load_model(args.model, args.revision, args.max_length)
    # Refuse any loader-specific template changes after preparing assistant-only labels.
    if runtime_tokenizer.get_vocab() != tokenizer.get_vocab() or (
            runtime_tokenizer.chat_template != tokenizer.chat_template):
        raise ValueError('Runtime tokenizer differs from the pinned training tokenizer')
    model = fast.get_peft_model(model, r=args.rank, lora_alpha=args.rank,
        target_modules=TARGET_MODULES, lora_dropout=0, bias='none',
        use_gradient_checkpointing='unsloth', random_state=args.seed,
        max_seq_length=args.max_length)
    model.config.use_cache = False
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer requires a pad or EOS token')

    def collate(batch):
        length = max(len(row['input_ids']) for row in batch)
        return {key: torch.tensor([row[key] + [padding] * (length - len(row[key])) for row in batch])
                for key, padding in [('input_ids', tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]}

    class Checkpoints(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            paused = bool(args.pause_file and args.pause_file.exists())
            if paused or state.global_step >= steps:
                control.should_save = True
            if paused:
                control.should_training_stop = True
            progress.update(phase='train', step=state.global_step, pause_requested=paused,
                            step_updated_at=time.time())
            return control

        def on_save(self, training_args, state, control, **kwargs):
            complete_checkpoint(args.output / f'checkpoint-{state.global_step}', binding)
            progress.update(checkpoint_step=state.global_step)
            return control

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if logs:
                if any(key in logs and not math.isfinite(float(logs[key]))
                       for key in ('loss', 'grad_norm')):
                    raise ValueError('Training produced nonfinite loss or gradients; adapter not completed')
                progress.update(metrics={k: v for k, v in logs.items() if isinstance(v, (int, float))})
            return control

    settings = TrainingArguments(output_dir=str(args.output), max_steps=steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, warmup_steps=min(10, steps // 10),
        lr_scheduler_type='linear', weight_decay=0.01, bf16=True, fp16=False,
        optim='adamw_8bit', logging_steps=1, save_strategy='steps', save_steps=args.save_steps,
        logging_nan_inf_filter=False,
        save_total_limit=3, report_to='none', push_to_hub=False,
        seed=args.seed, data_seed=args.seed, dataloader_num_workers=0, disable_tqdm=True)
    trainer = Trainer(model=model, args=settings, train_dataset=encoded,
                      data_collator=collate, callbacks=[Checkpoints()])
    progress.update(phase='train')
    recovered_complete = bool(saved and saved.global_step >= steps)
    if recovered_complete:
        # The last checkpoint can finish before final metadata. Never take an extra step.
        set_peft_model_state_dict(model, load_file(str(checkpoint / 'adapter_model.safetensors')))
        trainer.state = saved
    else:
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    final_checkpoint = verified_checkpoint(args.output / f'checkpoint-{trainer.state.global_step}', binding)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    status = 'completed' if trainer.state.global_step >= steps else 'paused'
    metadata = {'schema_version': 1, 'status': status, 'model': args.model, 'revision': args.revision,
                'precision': 'bf16', 'global_step': trainer.state.global_step, 'max_steps': steps,
                'included_count': len(encoded), 'excluded_count': report['excluded_count'],
                'languages': report['languages'], 'verification': read_json(args.manifest)['verification'],
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'versions': versions,
                'resumed_from': checkpoint.name if checkpoint else None,
                'recovered_completed_checkpoint': recovered_complete, 'evaluated': False,
                'adapter_sha256': sha256(args.output / 'adapter_model.safetensors'),
                'adapter_config_sha256': sha256(args.output / 'adapter_config.json'),
                'checkpoint': final_checkpoint.name}
    atomic_json(args.output / 'training.json', metadata)
    progress.update(status=status, phase='saved', step=trainer.state.global_step)
    print(f'QWEN35 {status}: {trainer.state.global_step}/{steps} steps; saved locally.', flush=True)


def main():
    args = parser().parse_args()
    validate_args(args)
    rows, _ = load_verified(args.dataset, args.manifest)
    if args.check_data:
        print(f'PASS: {len(rows)} provenance-checked training rows; no model loaded', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = request_binding(args)
        if args.resume and read_json(args.output / 'request.json') != request:
            raise ValueError('Resume data, source, or recipe changed')
        atomic_json(args.output / 'request.json', request)
        with Progress(args.output) as progress:
            train(args, rows, progress)


if __name__ == '__main__':
    main()
