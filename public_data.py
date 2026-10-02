"""Prepare a bounded public Python repair pilot; execute dataset code only in Docker."""
import argparse
import ast
import copy
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from recursive_agent import grade
from train_adapter import atomic_json
from training_data import sha256

SOURCE = {
    'dataset': 'nvidia/OpenCodeInstruct', 'license': 'CC-BY-4.0',
    'attribution': 'NVIDIA, OpenCodeInstruct',
    'url': 'https://huggingface.co/datasets/nvidia/OpenCodeInstruct',
    'revision': '8f3ba5bafe4d6e8db46082cf7ae6741bc370604d',
    'file': 'data/train-00000-of-00050.parquet',
    'file_sha256': '342757e0c6b706c8f68cf0a867ead5c417d0273726453dbcfc934f5c3c6ca891',
}


def read_parquet(path):
    if sha256(path) != SOURCE['file_sha256']:
        raise ValueError('Public parquet file does not match the pinned source digest')
    from pyarrow.parquet import ParquetFile
    rows = next(ParquetFile(path).iter_batches(batch_size=1000)).to_pylist()
    return {'source': SOURCE, 'rows': rows}


def task_from_row(row):
    if not isinstance(row, dict) or not re.fullmatch(r'[a-f0-9]{32}', str(row.get('id', ''))):
        raise ValueError('Invalid public row ID')
    prompt, answer = row.get('input'), row.get('output')
    if (not isinstance(prompt, str) or not 1 <= len(prompt) <= 4096
            or not isinstance(answer, str) or not 1 <= len(answer) <= 12000):
        raise ValueError('Public example exceeds the short pilot limits')
    try:
        tests = json.loads(row['unit_tests'])
        statuses = json.loads(row['tests_execution_status'])
        score = float(row['average_test_score'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Invalid public test metadata') from error
    if (not isinstance(tests, list) or not 2 <= len(tests) <= 32 or score != 1
            or statuses != ['pass'] * len(tests)):
        raise ValueError('Require at least two tests and complete passing metadata')
    chunks = re.findall(r'```(?:python)?[ \t]*\n(.*?)```', answer, re.S)
    source = chunks[0].strip()+'\n' if len(chunks) == 1 else answer.strip()+'\n'
    try:
        tree = ast.parse(source)
        if (not any(isinstance(n, ast.FunctionDef) for n in tree.body)
                or any(not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign)) for n in tree.body)):
            raise ValueError('Pilot supports standalone Python functions only')
        decoded = []
        for text in tests:
            if not isinstance(text, str) or len(text) > 4000:
                raise ValueError('Invalid public test source')
            try:
                parsed = ast.parse(text)
            except SyntaxError:
                # This dataset also stores JSON-escaped Python chunks inside the test list.
                text = json.loads('"'+text+'"')
                parsed = ast.parse(text)
            if not any(isinstance(n, ast.Assert) for n in ast.walk(parsed)):
                raise ValueError('Public test has no assertion')
            decoded.append(text.strip())
        if sum(map(len, decoded)) > 12000:
            raise ValueError('Public test suite exceeds the pilot limit')
    except (SyntaxError, TypeError, ValueError) as error:
        raise ValueError('Unsupported Python solution or tests') from error
    broken = copy.deepcopy(tree)
    for node in broken.body:
        if isinstance(node, ast.FunctionDef):
            node.body = [ast.Raise(exc=ast.Call(func=ast.Name(id='NotImplementedError', ctx=ast.Load()),
                                              args=[], keywords=[]), cause=None)]
    checks = "exec(open('/workspace/solution.py').read(), globals())\n"+'\n'.join(decoded)+'\n'
    identity = hashlib.sha256((prompt+ast.unparse(tree)).encode()).hexdigest()
    return {'id': 'opencode-'+row['id'], 'repository': 'opencode-'+identity,
            'split': 'dev' if int(identity[:2], 16) % 5 == 0 else 'train',
            'prompt': prompt+'\nComplete solution.py. Run test_visible.py before and after the repair. Keep the tests unchanged.',
            'files': {'solution.py': ast.unparse(ast.fix_missing_locations(broken))+'\n',
                      'test_visible.py': checks+"print('VISIBLE_CHECKS_PASSED')\n"},
            'editable': ['solution.py'], 'checks': checks,
            'reference_patch': {'solution.py': source},
            'provenance': {**SOURCE, 'row_id': row['id']}}


def prepare(snapshot, output, limit=500):
    snapshot, output = Path(snapshot), Path(output)
    if not 1 <= limit <= 500 or snapshot.stat().st_size > 64 * 1024**2 or output.exists():
        raise ValueError('Use a bounded snapshot and a new output folder')
    value = json.loads(snapshot.read_text())
    if (not isinstance(value, dict) or value.get('source') != SOURCE
            or not isinstance(value.get('rows'), list) or len(value['rows']) > 2000):
        raise ValueError('Snapshot does not identify the pinned public source')
    candidates, seen, excluded = [], set(), Counter()
    for row in value['rows']:
        try:
            task = task_from_row(row)
        except ValueError:
            excluded['metadata_or_syntax'] += 1
            continue
        if task['repository'] in seen:
            excluded['duplicate'] += 1
            continue
        seen.add(task['repository'])
        candidates.append(task)
        if len(candidates) == limit:
            break

    def checked(task):
        try:
            return grade(task, task['reference_patch'])['passed'] and not grade(
                task, {'solution.py': task['files']['solution.py']})['passed']
        except Exception:
            return False

    tasks = []
    # ponytail: two isolated graders bound CPU/RAM use; increase only for a measured larger data run.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for task, passed in zip(candidates, pool.map(checked, candidates)):
            if passed:
                tasks.append(task)
            else:
                excluded['isolated_test_failure'] += 1
            print(f'Checked {len(tasks)+excluded["isolated_test_failure"]}/{len(candidates)}; kept {len(tasks)}', flush=True)
    if not tasks:
        raise ValueError('No public examples passed the isolated reference and broken-code checks')
    output.mkdir(parents=True)
    atomic_json(output/'tasks.json', tasks)
    manifest = {'source': SOURCE, 'snapshot_sha256': sha256(snapshot),
                'tasks_sha256': sha256(output/'tasks.json'), 'rows': len(tasks),
                'splits': dict(Counter(t['split'] for t in tasks)), 'excluded': dict(excluded),
                'scope': 'Provided public tests rerun in isolated containers. Not a proof of general correctness. Original project holdout unused.'}
    atomic_json(output/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--snapshot', type=Path)
    inputs.add_argument('--parquet', type=Path, help='Pinned shard; requires pyarrow==23.0.1 in a separate data environment')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=500)
    args = parser.parse_args()
    if args.parquet:
        args.snapshot = args.output.with_name(args.output.name+'.snapshot.json')
        if args.snapshot.exists() or args.output.exists():
            parser.error('Choose a new output folder and snapshot path')
        atomic_json(args.snapshot, read_parquet(args.parquet))
    print(json.dumps(prepare(args.snapshot, args.output, args.limit), indent=2))
