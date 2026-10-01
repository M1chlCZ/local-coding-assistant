"""Export locally graded RLM training traces; never export evaluation splits."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

MAX_CHARS = 65536
MAX_FILE_BYTES = 16 * 1024 * 1024
DEFAULT_TASKS = Path(__file__).parent / 'research' / 'rlm_tasks.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    if Path(path).stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f'File exceeds {MAX_FILE_BYTES} bytes: {path}')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def registry(path):
    value = read_json(path)
    tasks = value.get('tasks') if isinstance(value, dict) else value
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Task registry must contain tasks')
    indexed = {}
    for task in tasks:
        if (not isinstance(task, dict) or not isinstance(task.get('id'), str)
                or not task['id'] or task['id'] in indexed
                or not isinstance(task.get('repository'), str)
                or not task['repository'] or task.get('split') not in ('train', 'dev', 'holdout', 'custom')):
            raise ValueError('Invalid or duplicate registry task')
        indexed[task['id']] = task
    return indexed


def messages_valid(messages):
    if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
        raise ValueError('Invalid trace messages')
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                or message['role'] not in ('system', 'user', 'assistant')
                or not isinstance(message['content'], str)):
            raise ValueError('Messages require text role/content only')
    if sum(len(message['content']) for message in messages) > MAX_CHARS:
        raise ValueError('Trace exceeds character limit')


def manifest_path(path):
    return Path(str(path) + '.manifest.json')


def repair_calls(task, trace, patch):
    """Execute an inspect/fail/repair/retest trajectory for a regraded teacher patch."""
    from recursive_agent import Sandbox, grade, parse_patch
    patch = parse_patch(json.dumps(patch), task)
    if not grade(task, patch)['passed']:
        raise ValueError('Repair no longer passes the registered functional checks')
    first = next((call for call in trace if call.get('kind') == 'root'), None)
    if first is None:
        raise ValueError('Verified repair requires a root prompt')
    messages_valid(first['messages'])
    if first['messages'][-1]['role'] != 'user':
        raise ValueError('Root prompt must end with a user message')
    if 'test_visible.py' not in task['files'] or 'test_visible.py' in task['editable']:
        raise ValueError('Tested repairs require an immutable visible test file')
    context = {'files': task['files'], 'editable': task['editable'], 'task': task['prompt']}
    test = ("import subprocess,sys\n"
            "tested = subprocess.run([sys.executable, '-B', '-I', '-c', "
            + repr("import sys;sys.path.insert(0,'/workspace');exec(open('/workspace/test_visible.py').read())")
            + "], capture_output=True, text=True, timeout=10)\n"
            "print('Test exit:', tested.returncode)\nprint(tested.stdout+tested.stderr)\n")
    repair = ("from pathlib import Path\npatch = " + repr(patch)
              + "\nfor name,source in patch.items():\n    Path('/workspace', name).write_text(source)\n"
              + test + "assert tested.returncode == 0\n")
    codes = ['print(context)', test, repair,
             "answer['content'] = patch\nanswer['ready'] = True"]
    environment = Sandbox(context, allow_queries=False)
    calls = []
    messages = list(first['messages'])
    try:
        for index, code in enumerate(codes):
            compile(code, '<verified-repair>', 'exec')
            response = '```repl\n'+code+'\n```'
            calls.append((list(messages), response))
            result = environment.execute_code(code)
            if index == 1 and environment_test_passed(result):
                raise ValueError('Visible tests do not reproduce the defect')
            if index == 2 and not environment_test_passed(result):
                raise ValueError('Teacher patch failed the visible retest')
            if index in (0, 3) and result.stderr:
                raise ValueError('Derived repair trajectory failed to execute')
            messages += [{'role':'assistant','content':response},
                         {'role':'user','content':'REPL output:\n\n'+result.stdout+result.stderr},
                         {'role':'user','content':f'Turn {index+2}/8:'}]
    finally:
        environment.cleanup()
    return calls


def environment_test_passed(result):
    return not result.stderr and 'Test exit: 0' in result.stdout and 'VISIBLE_CHECKS_PASSED' in result.stdout


def export(reports, tasks_path, output, repairs=False):
    output = Path(output)
    companion = manifest_path(output)
    if output.exists() or companion.exists():
        raise FileExistsError('Dataset or manifest already exists; choose a new filename')
    known = registry(tasks_path)
    digest = sha256(tasks_path)
    rows, sources, seen, excluded = [], [], set(), Counter()
    for path in reports:
        report = read_json(path)
        if (not isinstance(report, dict) or report.get('schema_version') != 1
                or report.get('tasks_sha256') != digest or report.get('split') != 'train'
                or not isinstance(report.get('tasks'), list)):
            raise ValueError('Only matching train evaluator reports are accepted')
        sources.append({'name': Path(path).name, 'sha256': sha256(path)})
        for task in report['tasks']:
            original = known.get(task.get('id')) if isinstance(task, dict) else None
            if original is None or any(task.get(key) != original[key] for key in ('repository', 'split')):
                raise ValueError('Report task does not match the pinned registry')
            if original['split'] != 'train':
                excluded['split'] += 1
                continue
            if task.get('mode') != 'rlm' or task.get('passed') is not True:
                excluded['failed_or_baseline'] += 1
                continue
            trace = task.get('trace')
            if not isinstance(trace, list) or not 1 <= len(trace) <= 128:
                raise ValueError('Passed RLM task has no bounded trace')
            if repairs:
                calls = repair_calls(original, trace, task.get('patch'))
            else:
                calls = []
            for call in ([] if repairs else trace):
                if not isinstance(call, dict):
                    raise ValueError('Malformed trace call')
                if call.get('kind') not in ('root', 'subcall'):
                    raise ValueError('Trace must distinguish root calls from subcalls')
                if call['kind'] == 'subcall':
                    excluded['subcall'] += 1
                    continue
                messages_valid(call.get('messages'))
                if call['messages'][-1]['role'] != 'user':
                    raise ValueError('Trace prompt must end with a user message')
                response = call.get('response')
                if not isinstance(response, str) or not response.strip():
                    raise ValueError('Missing assistant response')
                if any(type(call.get(key)) is not int or call[key] < 0 for key in ('input_tokens', 'output_tokens')):
                    raise ValueError('Invalid token counts')
                calls.append((call['messages'], response))
            for prompt, response in calls:
                messages = prompt + [{'role': 'assistant', 'content': response}]
                messages_valid(messages)
                identity = json.dumps(messages, sort_keys=True, ensure_ascii=False)
                if identity in seen:
                    excluded['duplicate'] += 1
                    continue
                seen.add(identity)
                rows.append({'messages': messages, 'task_id': original['id'],
                             'repository': original['repository'], 'tasks_sha256': digest,
                             'example_kind': 'executed_verified_repair' if repairs else 'root_trace'})
    if not rows:
        raise ValueError('No successful train-split RLM traces to export')
    content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    if len(content.encode('utf-8')) > MAX_FILE_BYTES:
        raise ValueError('Dataset exceeds file limit; export fewer reports')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)
    manifest = {'schema_version': 1, 'split': 'train', 'tasks_sha256': digest,
                'dataset_sha256': sha256(output), 'rows': len(rows), 'sources': sources,
                'excluded': dict(excluded),
                'example_kind': 'executed_verified_repair' if repairs else 'root_trace',
                'trust': 'Local evaluator reports are inputs, not cryptographic proof of correct grading.'}
    with companion.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load_verified(dataset, tasks_path=DEFAULT_TASKS):
    if Path(dataset).stat().st_size > MAX_FILE_BYTES:
        raise ValueError('Dataset exceeds file limit')
    manifest = read_json(manifest_path(dataset))
    if not isinstance(manifest, dict):
        raise ValueError('Invalid dataset manifest')
    digest = sha256(tasks_path)
    if (manifest.get('schema_version') != 1 or manifest.get('split') != 'train'
            or manifest.get('tasks_sha256') != digest or manifest.get('dataset_sha256') != sha256(dataset)):
        raise ValueError('Dataset or task registry does not match its manifest')
    known, rows = registry(tasks_path), []
    for line in Path(dataset).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError('Invalid dataset row')
        original = known.get(row.get('task_id'))
        if (original is None or original['split'] != 'train' or row.get('repository') != original['repository']
                or row.get('tasks_sha256') != digest):
            raise ValueError('Dataset row is not a registered training task')
        messages_valid(row.get('messages'))
        if (len(row['messages']) < 2 or row['messages'][-1]['role'] != 'assistant'
                or not row['messages'][-1]['content'].strip() or row['messages'][-2]['role'] != 'user'):
            raise ValueError('Dataset requires final assistant response and preceding user prompt')
        rows.append(row)
    if not rows or manifest.get('rows') != len(rows):
        raise ValueError('Dataset row count does not match its manifest')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--tasks', type=Path, default=DEFAULT_TASKS)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(export(args.reports, args.tasks, args.output), indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
