import hashlib
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
name, label, keep, tag = sys.argv[1:]
path = root / '.cache/models' / name
with path.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
config = json.loads((root / 'manifest.json').read_text())
config['model'] = {'name': name, 'label': label, 'url': None,
                   'size': path.stat().st_size, 'sha256': digest, 'license': 'Apache-2.0',
                   'source_repo': 'Qwen/Qwen3.6-35B-A3B',
                   'source_revision': '995ad96eacd98c81ed38be0c5b274b04031597b0',
                   'custom_derivative': float(keep) < 1, 'keep_ratio': float(keep),
                   'quantization': 'Q4_K_M', 'importance_matrix': False,
                   'converter_commit': '7fe450e19305b828c199d602c23a8337aaa1f03b'}
config['defaults'] = {'context': 8192, 'cpu_moe': {1.0: 20, 0.75: 8, 0.5: 0}[float(keep)], 'threads': 12}
if float(keep) < 1:
    config['model']['criterion'] = 'reap'
    for key, source in [('calibration_sha256', root / 'research/calib.jsonl'),
                        ('windows_sha256', root / '.cache/research/calibrated/windows.json'),
                        ('selection_sha256', root / f'.cache/research/calibrated/keep_reap_{keep}.json')]:
        config['model'][key] = hashlib.sha256(source.read_bytes()).hexdigest()
out = root / f'research/manifest-{tag}.json'
out.write_text(json.dumps(config, indent=2), encoding='utf-8')
print(out, path.stat().st_size, digest, flush=True)
