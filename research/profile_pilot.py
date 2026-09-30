"""Profile one deterministic token window per calibration sample with disk offload."""
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
from transformers import AutoTokenizer
from moep.adapters import get_adapter
from moep.data import read_jsonl, tokenize_rows
from moep.loader import load_offloaded
from moep.profiler import pack_token_stream, profile, write_summary

ROOT = Path(__file__).resolve().parents[1]

def main():
    base = ROOT / '.cache/research/base'
    out = ROOT / '.cache/research/calibrated'
    out.mkdir(parents=True, exist_ok=True)
    data = ROOT / 'research/calib.jsonl'
    manifest = json.loads((ROOT / 'research/calibration_manifest.json').read_text())
    if hashlib.sha256(data.read_bytes()).hexdigest() != manifest['sha256']:
        raise RuntimeError('Calibration checksum does not match')
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    rows = read_jsonl(data)
    tokens = tokenize_rows(tokenizer, rows)
    assert len(tokens) == len(rows), 'Every calibration sample must render'
    rng = random.Random(17)
    windows, spans = [], []
    for row, ids in zip(rows, tokens):
        start = rng.randrange(max(1, len(ids) - 256 + 1))
        windows.append(ids[start:start + 256])
        spans.append({'source': row['source'], 'row_index': row['row_index'],
                      'token_offset': start, 'tokens': len(windows[-1])})
    batches = pack_token_stream(windows, 256, tokenizer.eos_token_id)
    torch.save(batches, out / 'calibration_tokens.pt')
    details = {'seed': 17, 'calibration_sha256': manifest['sha256'], 'samples': spans,
               'sequence_length': 256, 'sequences': len(batches), 'batch_size': 8,
               'scope': 'One random window per source sample; small calibration pilot, not full paper reproduction.'}
    (out / 'windows.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    print(f'Calibration: {len(rows)} samples, {len(batches)} sequences, {batches.numel()} tokens', flush=True)
    started = time.time()
    model = load_offloaded(base, 10, ROOT / '.cache/research/offload')
    adapter = get_adapter(json.loads((base / 'config.json').read_text()))
    stats, layers = profile(model, adapter, batches, batch_size=8, log_every=1)
    summary = write_summary(out / 'profile.stats.pt', stats, layers,
                            {'model': str(base), 'calib': str(data), 'seq_len': 256,
                             'n_samples': len(rows), 'started_unix': started,
                             'ended_unix': time.time(), 'max_gpu_gib': 10, 'batch_size': 8})
    if not all(summary['stats_finite'].values()):
        raise RuntimeError('Non-finite scores: refusing to select experts')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
