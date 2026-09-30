"""Reconstruct the recorded calibration pilot from Dataset Viewer snapshots."""
import hashlib
import json
import random
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    manifest = json.loads((ROOT / 'calibration_manifest.json').read_text())
    rows = []
    for source in manifest['sources']:
        indices = source['row_indices']
        assert indices == list(range(indices[0], indices[0] + len(indices)))
        query = urllib.parse.urlencode({'dataset': source['repo'], 'config': source['config'],
                                        'split': source['split'], 'offset': indices[0], 'length': len(indices)})
        raw = subprocess.check_output([shutil.which('curl.exe') or 'curl', '--fail', '--silent',
                                       '--show-error', 'https://datasets-server.huggingface.co/rows?' + query])
        data = json.loads(raw)
        for wrapper in data['rows']:
            if wrapper.get('truncated_cells'):
                raise RuntimeError('Dataset Viewer truncated a calibration sample')
            row = wrapper['row']
            if source['repo'].endswith('evol-codealpaca-v1'):
                messages = [{'role': 'user', 'content': row['instruction']},
                            {'role': 'assistant', 'content': row['output']}]
            elif source['repo'].endswith('hermes-function-calling-v1'):
                messages = [{'role': {'human': 'user', 'gpt': 'assistant'}.get(m['from'], m['from']),
                             'content': m['value']} for m in row['conversations']]
            else:
                messages = row['messages']
            messages = [{'role': m['role'], 'content': m['content']} for m in messages
                        if m['role'] in ['system', 'user', 'assistant'] and m['content']]
            if len(messages) >= 2:
                rows.append({'messages': messages, 'source': source['repo'], 'row_index': wrapper['row_idx']})
    random.Random(manifest['seed']).shuffle(rows)
    encoded = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
    if hashlib.sha256(encoded).hexdigest() != manifest['sha256']:
        raise RuntimeError('Calibration snapshot changed; refusing to replace the recorded pilot')
    (ROOT / 'calib.jsonl').write_bytes(encoded)
    print(f'Verified calibration: {len(rows)} rows')

if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit(f'Error: {error}')
