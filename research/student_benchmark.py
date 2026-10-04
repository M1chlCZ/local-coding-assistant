"""Matched direct HumanEval generation for the local NF4 student and its adapter."""
import argparse
import fcntl
import gzip
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark import assemble_solution
from evaluation import IMAGE, chat, check_code, request
from launcher import health, verify
from train_adapter import atomic_json, verified_checkpoint
from training_data import sha256

MODEL = 'Qwen/Qwen3-4B'
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
SYSTEM = ('You are an English coding assistant. Return complete, concise Python source only, '
          'including needed imports and supplied helper functions. '
          'Do not repeat docstrings, examples, or reasoning.')
SOURCES = ('research/student_benchmark.py', 'research/adapter_eval_server.py',
           'benchmark.py', 'evaluation.py', 'launcher.py', 'train_adapter.py', 'training_data.py')


def paused_worker_pid(action, status, owner, start, group, environment, command, controller):
    if (action != 'pause' or status != 'paused' or not owner.get('forward')
            or str(owner['start']) != start or group != owner['pid']
            or ('LCA_CONTINUOUS_OWNER='+controller).encode() not in environment
            or not any(Path(part).name == 'learning_session.py' for part in command)):
        raise ValueError('Learning worker is active or its process ownership is not established')
    return owner['pid']


def release_paused_worker(path):
    """End only the identified idle worker; saved Pause and checkpoints stay intact."""
    from continuous_learning import Controller, read
    controller = Controller(path)
    controller.validate()
    snapshot = controller.snapshot()
    marker = read(controller.path/'command.json')
    if snapshot['desired'] != 'pause' or snapshot['status'] != 'paused':
        raise ValueError('Select Pause and wait for Paused before the benchmark handoff')
    owner = read(controller.path/'process.json')
    if not owner:
        return
    proc = Path('/proc')/str(owner['pid'])
    try:
        start = proc.joinpath('stat').read_text().split(') ', 1)[1].split()[19]
        environment = proc.joinpath('environ').read_bytes().split(b'\0')
        command = proc.joinpath('cmdline').read_bytes().decode().split('\0')
        pid = paused_worker_pid(snapshot['desired'], snapshot['status'], owner, start,
                                os.getpgid(owner['pid']), environment, command, str(controller.path))
    except (FileNotFoundError, ProcessLookupError):
        return
    if read(controller.path/'command.json') != marker:
        raise ValueError('Learning control changed before benchmark handoff')
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic()+30
    while (controller.path/'process.json').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Paused learning worker did not finish its handoff')
        time.sleep(0.2)


def load_tasks():
    metadata = json.loads((ROOT/'research/humaneval_manifest.json').read_text())
    source = ROOT/'research/HumanEval.jsonl.gz'
    verify(source, metadata)
    with gzip.open(source, 'rt', encoding='utf-8') as stream:
        tasks = [json.loads(line) for line in stream]
    tasks.sort(key=lambda t: int(t['task_id'].split('/')[1]))
    if len(tasks) != 164 or len({t['task_id'] for t in tasks}) != 164:
        raise ValueError('The complete pinned HumanEval registry requires 164 distinct tasks')
    return metadata, tasks


def load_report(path, binding, tasks):
    report = json.loads(path.read_text()) if path.exists() else {'binding': binding, 'results': []}
    if report.get('binding') != binding:
        raise ValueError('Benchmark binding changed; use a separate output folder')
    ids = [t['task_id'] for t in tasks]
    if [r['task_id'] for r in report['results']] != ids[:len(report['results'])]:
        raise ValueError('Benchmark progress does not match the fixed task order')
    return report


def summarize(binding, tasks, reports):
    total = len(tasks)
    if any(len(reports[name]['results']) != total for name in ('base', 'adapter')):
        raise ValueError('Both models must complete every task before comparison')
    flags = {name: {r['task_id']: r.get('passed') is True for r in report['results']}
             for name, report in reports.items()}
    gained = [t['task_id'] for t in tasks if flags['adapter'][t['task_id']] and not flags['base'][t['task_id']]]
    lost = [t['task_id'] for t in tasks if flags['base'][t['task_id']] and not flags['adapter'][t['task_id']]]
    models = {}
    for name, report in reports.items():
        rows = report['results']
        seconds = sum(r['elapsed_s'] for r in rows)
        tokens = sum(r['output_tokens'] for r in rows)
        models[name] = {'passed': sum(flags[name].values()), 'total': total,
                        'pass_at_1': sum(flags[name].values())/total,
                        'model_answer_seconds': round(seconds, 3),
                        'median_model_answer_seconds': round(statistics.median(r['elapsed_s'] for r in rows), 3),
                        'output_tokens': tokens,
                        'output_tokens_per_request_second': round(tokens/seconds, 3) if seconds else None,
                        'outputs_at_token_cap': sum(r['output_tokens'] >= binding['max_output_tokens'] for r in rows),
                        'pass_flags': flags[name]}
    return {'completed': True, 'binding': binding, 'models': models,
            'adapter_vs_base': {'gained': gained, 'lost': lost,
                                'higher_score': models['adapter']['passed'] > models['base']['passed'],
                                'no_regression': not lost},
            'scope': 'Full 164-task HumanEval with original tests. Direct greedy first attempt, no repair tools or retries. Not HumanEval+ or an agent benchmark.',
            'limitations': ['Public benchmark overlap with pretraining or public training sources is unknown.',
                           'Original HumanEval tests have limited edge-case coverage.',
                           'Request speed includes prompt processing and HTTP overhead; it is not isolated decode speed.',
                           'These tasks must not become new training targets or repeated checkpoint selection checks.']}


def run(adapter, output, paused_controller=None):
    metadata, tasks = load_tasks()
    adapter = adapter.resolve()
    verified_checkpoint(adapter, json.loads((adapter.parent/'run.json').read_text()))
    config = json.loads((adapter/'adapter_config.json').read_text())
    if config.get('base_model_name_or_path') != MODEL:
        raise ValueError('Adapter requires a different base model')
    binding = {'model': MODEL, 'revision': REVISION, 'quantization': 'NF4',
               'adapter_sha256': sha256(adapter/'adapter_model.safetensors'),
               'adapter_config_sha256': sha256(adapter/'adapter_config.json'),
               'dataset': {k: metadata[k] for k in ('repo', 'revision', 'url', 'sha256', 'license')},
               'tasks': len(tasks), 'temperature': 0, 'seed': 42, 'thinking': False,
               'max_output_tokens': 1024, 'attempts_per_task': 1,
               'execution_timeout_seconds': 20, 'container_image': IMAGE,
               'system_prompt': SYSTEM, 'sources': {name: sha256(ROOT/name) for name in SOURCES}}
    output.mkdir(parents=True, exist_ok=True)
    prior = json.loads((output/'binding.json').read_text()) if (output/'binding.json').exists() else binding
    if prior != binding:
        raise ValueError('Benchmark binding changed; use a separate output folder')
    atomic_json(output/'binding.json', binding)
    reports = {name: load_report(output/f'{name}.json', binding, tasks) for name in ('base', 'adapter')}
    if (output/'summary.json').exists():
        summarize(binding, tasks, reports)
        print('Benchmark already completed; no repeat generation', flush=True)
        return
    subprocess.run(['docker', 'image', 'inspect', IMAGE], check=True, capture_output=True, timeout=30)
    audit = json.loads((output/'reference-audit.json').read_text()) if (output/'reference-audit.json').exists() else {'binding': binding, 'results': []}
    if audit.get('binding') != binding or [r['task_id'] for r in audit['results']] != [t['task_id'] for t in tasks[:len(audit['results'])]]:
        raise ValueError('Reference audit binding changed')
    for task in tasks[len(audit['results']):]:
        atomic_json(output/'status.json', {'status': 'running', 'phase': 'reference-audit', 'completed': len(audit['results']), 'total': len(tasks)})
        checks = task['test']+'\ncheck('+task['entry_point']+')'
        row = {'task_id': task['task_id'], **check_code(task['prompt']+task['canonical_solution'], checks)}
        if not row['passed']:
            raise RuntimeError(f"Reference audit failed for {task['task_id']}: {row['feedback']}")
        audit['results'].append(row)
        atomic_json(output/'reference-audit.json', audit)
    if paused_controller:
        release_paused_worker(paused_controller)
    lock = (ROOT/'.cache/learning/gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    process = None
    try:
        for port in (8080, 8090):
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', port)) == 0:
                    raise RuntimeError('Model port occupied; pause learning and close chat before benchmarking')
        with (output/'server.log').open('ab') as log:
            process = subprocess.Popen([str(ROOT/'.cache/train-env/bin/python'),
                str(ROOT/'research/adapter_eval_server.py'), '--adapter', str(adapter)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+180
        while not health(8090):
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Student benchmark server failed to start')
            time.sleep(1)
        for name in ('base', 'adapter'):
            request('http://127.0.0.1:8090', '/mode', {'mode': 'base' if name == 'base' else 'adapter'})
            report = reports[name]
            for task in tasks[len(report['results']):]:
                atomic_json(output/'status.json', {'status': 'running', 'phase': name,
                    'completed': len(report['results']), 'total': len(tasks), 'task_id': task['task_id']})
                response, seconds = chat('http://127.0.0.1:8090', [
                    {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task['prompt']}], max_tokens=1024)
                content = response['choices'][0]['message'].get('content') or ''
                checked = check_code(assemble_solution(task, content), task['test']+'\ncheck('+task['entry_point']+')')
                if checked['exit_code'] in (125, 126, 127):
                    raise RuntimeError('Container infrastructure failed; benchmark remains incomplete')
                row = {'task_id': task['task_id'], 'content': content, 'elapsed_s': seconds,
                       'output_tokens': response['usage']['completion_tokens'], **checked}
                report['results'].append(row)
                atomic_json(output/f'{name}.json', report)
                print(name, task['task_id'], 'PASS' if row['passed'] else 'FAIL', seconds, flush=True)
        atomic_json(output/'summary.json', summarize(binding, tasks, reports))
        atomic_json(output/'status.json', {'status': 'completed', 'completed': len(tasks), 'total': len(tasks)})
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paused-controller', type=Path,
                        help='Release the identified idle learning worker after a saved Pause')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        run(args.adapter, args.output, args.paused_controller)
    except BaseException as error:
        atomic_json(args.output/'status.json', {'status': 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                                               'detail': f'{type(error).__name__}: {error}'})
        raise
