"""Pinned, locally checked teacher functions. Reference answers stay out of validation prompts."""
import ast
import hashlib
import json
from pathlib import Path
import re
import uuid

from focused_data import benchmark_exclusions, download, shingles
from polyglot_runtime import check_program, image
from qwen35_train import atomic_json, sha256

ROOT = Path(__file__).resolve().parent
DATASET = 'nvidia/OpenCodeInstruct'
REVISION = '8f3ba5bafe4d6e8db46082cf7ae6741bc370604d'
FILE = 'data/train-00000-of-00050.parquet'
DIGEST = '342757e0c6b706c8f68cf0a867ead5c417d0273726453dbcfc934f5c3c6ca891'
SYSTEM = ('You are an English coding assistant. Return complete source for the requested function '
          'and needed imports/helpers. No tests, main, examples or reasoning. Use python.')


def convert(row, exclusions, excluded_names=frozenset()):
    """Publisher scores are only a prefilter; execution is still required locally."""
    try:
        if float(row['average_test_score']) != 1:
            return None
        tests = json.loads(row['unit_tests'])
        statuses = json.loads(row['tests_execution_status'])
        if (not isinstance(tests, list) or not 5 <= len(tests) <= 50
                or statuses != ['pass'] * len(tests) or any(not isinstance(t, str) for t in tests)):
            return None
        output = row['output'].strip()
        fences = re.findall(r'```(?:python|py)?\s*\n(.*?)```', output, re.S)
        if '```' in output and len(fences) != 1:
            return None
        answer = (fences[0] if fences else output).strip()
        if len(row['input']) + len(answer) > 14000 or sum(map(len, tests)) > 30000:
            return None
        tree = ast.parse(answer)
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if len(functions) != 1 or any(not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)) for n in tree.body):
            return None
        name = functions[0].name
        if name in excluded_names:
            return None
        for test in tests:
            nodes = ast.parse(test).body
            if not nodes or any(not isinstance(n, ast.Assert) for n in nodes):
                return None
            if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
                       for n in ast.walk(ast.parse(test))):
                return None
        if shingles(row['input'] + ' ' + answer) & exclusions:
            return None
        source_sha = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        return dict(id='opencodeinstruct-' + source_sha[:24], family=name.lower(),
                    prompt=row['input'], answer=answer, tests='\n'.join(tests),
                    provenance=dict(dataset=DATASET, revision=REVISION, source_row_sha256=source_sha))
    except (KeyError, TypeError, ValueError, SyntaxError):
        return None


def split_for(entry):
    # Same function name, including description variants, always stays in one split.
    value = hashlib.sha256(('teacher-v1:' + entry['family']).encode()).hexdigest()
    return 'dev' if int(value[:8], 16) % 5 == 0 else 'train'


def task(entry, split):
    imports = '\n'.join(ast.unparse(node) for node in ast.parse(entry['answer']).body
                        if isinstance(node, (ast.Import, ast.ImportFrom)))
    return dict(id=entry['id'], repository='teacher-' + entry['family'], split=split,
                language='python', kind='teacher_function', prompt=entry['prompt'],
                tests=entry['tests'], test_imports=imports, provenance=entry['provenance'])


def messages(task):
    return [dict(role='system', content=SYSTEM), dict(role='user', content=task['prompt'])]


def check(task, answer):
    # Test fixtures must not supply imports to the candidate function's globals.
    marker = 'TEACHER_CHECKS_COMPLETED_'+uuid.uuid4().hex
    tests = (f'solution_scope = {{}}\nexec({answer!r}, solution_scope)\n'
               'test_scope = dict(solution_scope)\n'
               f'exec({task.get("test_imports", "")!r}, test_scope)\n'
               f'exec({task["tests"]!r}, test_scope)\n')
    program = ('import contextlib,io\nwith contextlib.redirect_stdout(io.StringIO()):\n'+
               ''.join(' '+line+'\n' for line in tests.splitlines())+f'print({marker!r})\n')
    # The outer runner must observe test completion, not merely a successful early exit.
    return check_program('python', program, [{'input':'','output':marker}])


def verify(entry):
    value = task(entry, split_for(entry))
    return check(value, entry['answer'])['passed'] and not check(value, '')['passed']


def load_round(folder):
    path = folder/'teacher.json'
    if not path.exists():
        return None
    prepared = json.loads((folder/'prepare.json').read_text())
    if prepared.get('teacher_sha256') != sha256(path):
        raise ValueError('Prepared teacher data changed')
    return json.loads(path.read_text())


def prepare_round(session, folder, train_count, dev_count, progress):
    """Resume CPU verification without regenerating answers or touching model weights."""
    from pyarrow.parquet import ParquetFile
    path = download(f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{FILE}',
                    ROOT/'.cache/teacher-source/opencodeinstruct-00000.parquet', DIGEST)
    exclusions, benchmarks = benchmark_exclusions()
    # Exclude benchmark function names as well as verbatim 13-word overlaps.
    from research.polyglot_benchmark import load_tasks
    _, benchmarks_tasks = load_tasks()
    excluded_names = {t['entry_point'] for tasks in benchmarks_tasks.values() for t in tasks}
    binding = dict(source_sha256=DIGEST, source_revision=REVISION, benchmarks=benchmarks,
                   image=image(), sources={name:sha256(ROOT/name) for name in
                   ('teacher_data.py','focused_data.py','polyglot_runtime.py','evaluation.py')},
                   train_count=train_count, dev_count=dev_count)
    used, previous_end = set(), 0
    other_words = {split:set() for split in ('train','dev')}
    for old in sorted(session.glob('round-*/teacher.json')):
        if old.parent.name >= folder.name:
            continue
        data = load_round(old.parent)
        used.update(data['families'])
        previous_end = max(previous_end, data['cursor'])
        for key, split in (('train','train'),('validation','dev')):
            for entry in data[key]:
                other_words[split].update(shingles(entry['prompt']+' '+entry['answer']))
    draft = folder/'teacher-preparation.json'
    saved = json.loads(draft.read_text()) if draft.exists() else dict(
        binding=binding, cursor=previous_end, train=[], validation=[], families=[], rejected=0)
    if saved['binding'] != binding:
        raise ValueError('Teacher preparation source or checks changed')
    used.update(saved['families'])
    # Reject cross-split near copies within this batch as well as exact family variants.
    for key, split in (('train','train'),('validation','dev')):
        for entry in saved[key]:
            other_words[split].update(shingles(entry['prompt']+' '+entry['answer']))
    index = 0
    for batch in ParquetFile(path).iter_batches(batch_size=512):
        for raw in batch.to_pylist():
            index += 1
            if index <= saved['cursor']:
                continue
            if len(saved['train']) >= train_count and len(saved['validation']) >= dev_count:
                break
            entry = convert(raw, exclusions, excluded_names)
            key = 'train' if entry and split_for(entry) == 'train' else 'validation'
            cap = train_count if key == 'train' else dev_count
            split = 'train' if key == 'train' else 'dev'
            words = shingles(entry['prompt']+' '+entry['answer']) if entry else set()
            eligible = (entry and entry['family'] not in used and len(saved[key]) < cap
                        and not words & other_words['dev' if split == 'train' else 'train'])
            if eligible:
                progress(folder, 'verify-teacher', len(saved['train'])+len(saved['validation']),
                         train_count+dev_count, task=entry['id'], language='python')
            if eligible and verify(entry):
                saved[key].append(entry); saved['families'].append(entry['family'])
                used.add(entry['family']); other_words[split].update(words)
            else:
                saved['rejected'] += 1
            saved['cursor'] = index
            # All accepted checks are durable; rejected rows may safely be retried after Pause.
            if eligible or index % 50 == 0:
                atomic_json(draft, saved)
        if len(saved['train']) >= train_count and len(saved['validation']) >= dev_count:
            break
    if len(saved['train']) < train_count or len(saved['validation']) < dev_count:
        atomic_json(draft, saved)
        raise ValueError('Pinned teacher shard exhausted; add a reviewed source before more teacher training')
    saved.update(license='CC-BY-4.0', attribution='NVIDIA, OpenCodeInstruct (Ahmad et al., 2025)',
                 source_url=f'https://huggingface.co/datasets/{DATASET}/tree/{REVISION}',
                 limitations=['Synthetic reference tests can be wrong; local passing checks are not proof of correctness.',
                              'Name and text exclusions cannot prove absence of semantic or pretraining overlap.',
                              'This teacher source covers Python only; other languages use verified replay.'])
    atomic_json(folder/'teacher.json', saved)
    return saved


def training_rows(folder):
    data = load_round(folder)
    if not data:
        return []
    return [dict(task_id=e['id'], split='train', language='python',
                 messages=messages(task(e,'train'))+[dict(role='assistant',content=e['answer'])],
                 verification=dict(kind='local_execution',passed=True), provenance=e['provenance'])
            for e in data['train'] if split_for(e) == 'train']
