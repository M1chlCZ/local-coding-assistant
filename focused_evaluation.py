"""Saved one-attempt functional audits for the finite local model comparison."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from evaluation import chat_batch, request
from launcher import health
from research.polyglot_benchmark import checked, load_tasks
from train_adapter import atomic_json
from training_data import sha256

ROOT = Path(__file__).resolve().parent
QWEN35 = 'Qwen/Qwen3.5-4B'
QWEN35_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
PROMPT = 'Return complete source for the supplied function and needed imports/helpers. No tests, main, examples or reasoning.'


def summarize(binding, tasks, reports):
    languages = {}
    for language, expected in tasks.items():
        rows = reports[language]
        if [r['id'] for r in rows] != [t['id'] for t in expected]:
            raise ValueError('A complete audit requires every registered task in fixed order')
        if any(type(r.get('passed')) is not bool or r.get('error') for r in rows):
            raise ValueError('Audit contains an ungraded infrastructure failure')
        passed = sum(r['passed'] for r in rows)
        languages[language] = {'passed': passed, 'total': len(rows), 'pass_at_1': passed/len(rows)}
    return {'completed': True, 'binding': binding, 'languages': languages,
        'passed': sum(v['passed'] for v in languages.values()),
        'total': sum(v['total'] for v in languages.values()),
        'macro_pass_at_1': sum(v['pass_at_1'] for v in languages.values())/len(languages),
        'scope': ('Small reserved repository repair pilot; not an agent benchmark or a general quality estimate'
                  if binding.get('suite') == 'repairs' else
                  'HumanEval/MultiPL-E function tests, one greedy attempt; not a repository or agent benchmark'),
        'purpose': 'Report only. No audit task or answer enters training or recipe selection.'}


def run(args):
    if not os.environ.get('LCA_FOCUSED_OWNER'):
        raise ValueError('Run through the finite worker so it owns the GPU and Pause control')
    if args.suite == 'repairs':
        from focused_replay import grade
        selection = json.loads(args.selection.read_text())
        if not selection.get('completed') or not selection.get('tasks'):
            raise ValueError('Prepare and freeze locally valid repository controls first')
        reserved = {r['task_id']: r for r in map(json.loads, args.repairs.read_text().splitlines())}
        tasks = {}
        for selected in selection['tasks']:
            row = reserved[selected['task_id']]
            # Never include the reference assistant answer or tests in the prompt.
            tasks.setdefault(row['language'], []).append({'id': row['task_id'],
                'messages': [m for m in row['messages'] if m['role'] != 'assistant']})
        if set(tasks) != {'go', 'typescript'}:
            raise ValueError('The repair pilot needs valid controls in both languages')
        metadata = {str(p.name): sha256(p) for p in (args.selection, args.repairs, args.replay_data)}
        check = lambda language, task, answer: grade(args.replay_data, args.selection, task['id'], answer)
        max_tokens = 2048
    else:
        metadata, tasks = load_tasks()
        check = checked
        max_tokens = 1024
    if args.smoke_limit:
        tasks = {l: rows[:args.smoke_limit] for l, rows in tasks.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    sources = ('focused_evaluation.py', 'research/polyglot_benchmark.py', 'polyglot_runtime.py',
               'evaluation.py', 'qwen35_server.py', 'qwen35_train.py', 'research/adapter_eval_server.py',
               'focused_replay.py', 'focused_data.py')
    from polyglot_runtime import image
    model = QWEN35 if args.engine == 'qwen35' else 'Qwen/Qwen3-4B'
    revision = QWEN35_REVISION if args.engine == 'qwen35' else '1cfa9a7208912126459214e8b04321603b3df60c'
    runtime_python=ROOT/('.cache/qwen35-env/bin/python' if args.engine == 'qwen35' else '.cache/train-env/bin/python')
    runtime_script=("from qwen35_train import runtime_versions;import json;print(json.dumps(runtime_versions()))"
        if args.engine == 'qwen35' else
        "import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ('torch','transformers','peft','bitsandbytes')}))")
    runtime=json.loads(subprocess.check_output([str(runtime_python),'-c',runtime_script],cwd=ROOT,text=True,timeout=30))
    binding = {'engine': args.engine, 'model': model, 'revision': revision,
        'mode': args.mode, 'precision': 'bf16' if args.engine == 'qwen35' else 'nf4',
        'adapter_sha256': sha256(args.adapter/'adapter_model.safetensors') if args.mode == 'adapter' else None,
        'datasets': metadata, 'container_image': image(), 'suite': args.suite,
        'prompt': 'Original issue and buggy hunks; complete unified patch' if args.suite == 'repairs' else PROMPT,
        'max_output_tokens': max_tokens, 'temperature': 0, 'thinking': False,
        'batch_size': args.batch_size, 'smoke_limit': args.smoke_limit, 'runtime': runtime,
        'sources': {n: sha256(ROOT/n) for n in sources}}
    previous = args.output/'binding.json'
    if previous.exists() and json.loads(previous.read_text()) != binding:
        raise ValueError('Audit model, sources or settings changed; use a new output folder')
    atomic_json(previous, binding)
    reports = {}
    for language in tasks:
        path = args.output/(language+'.json')
        reports[language] = json.loads(path.read_text()) if path.exists() else []
        if [r['id'] for r in reports[language]] != [t['id'] for t in tasks[language][:len(reports[language])]]:
            raise ValueError('Saved audit order differs from the fixed task list')
    summary_path = args.output/('smoke-summary.json' if args.smoke_limit else 'summary.json')
    if summary_path.exists():
        expected = summarize(binding, tasks, reports)
        if args.smoke_limit:
            expected.update(completed=False, smoke_completed=True)
        if json.loads(summary_path.read_text()) != expected:
            raise ValueError('Completed audit changed')
        return
    for port in (8080, 8090):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError('Model port occupied; close chat before the comparison')
    if args.engine == 'qwen35':
        command = [str(ROOT/'.cache/qwen35-env/bin/python'), str(ROOT/'qwen35_server.py'),
                   '--revision', revision, '--batch-size', str(min(args.batch_size, 2)),
                   '--output-limit', str(max_tokens)]
        if args.adapter:
            command += ['--adapter', str(args.adapter)]
    else:
        if not args.adapter:
            raise ValueError('The previous student bridge requires its intact local adapter')
        command = [str(ROOT/'.cache/train-env/bin/python'), str(ROOT/'research/adapter_eval_server.py'),
                   '--adapter', str(args.adapter), '--output-limit', str(max_tokens)]
    process = None
    try:
        with (args.output/'server.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+900
        while not health(8090):
            atomic_json(args.output/'progress.json', {'status': 'running', 'phase': 'load_model',
                'completed': sum(map(len, reports.values())), 'total': sum(map(len, tasks.values())),
                'heartbeat_at': time.time()})
            if process.poll() is not None or time.monotonic() > deadline:
                tail = (args.output/'server.log').read_text(errors='replace')[-4000:]
                raise RuntimeError('Model failed to load: '+tail)
            time.sleep(2)
        request('http://127.0.0.1:8090', '/mode', {'mode': args.mode}, timeout=30)
        for language, expected in tasks.items():
            rows = reports[language]
            for offset in range(len(rows), len(expected), args.batch_size):
                chunk = expected[offset:offset+args.batch_size]
                atomic_json(args.output/'progress.json', {'status': 'running', 'phase': 'evaluate',
                    'language': language, 'completed': sum(map(len, reports.values())),
                    'total': sum(map(len, tasks.values())), 'task': chunk[0]['id'], 'heartbeat_at': time.time()})
                messages = [task['messages'] for task in chunk] if args.suite == 'repairs' else [[
                    {'role': 'system', 'content': 'You are an English coding assistant. '+PROMPT+' Use '+language+'.'},
                    {'role': 'user', 'content': task['prompt']}] for task in chunk]
                responses, elapsed = chat_batch('http://127.0.0.1:8090', messages, max_tokens=max_tokens, timeout=600)
                answers = [r['choices'][0]['message'].get('content') or '' for r in responses]
                # Keep container checks on the signal-handling thread. Pause must unwind
                # each check's finally block and remove its container before releasing the GPU.
                results = [check(language, task, answer) for task, answer in zip(chunk, answers)]
                for task, response, answer, result in zip(chunk, responses, answers, results):
                    if result.get('exit_code') in (125, 126, 127):
                        raise RuntimeError('Compiler infrastructure failed')
                    rows.append({'id': task['id'], 'content': answer, 'elapsed_s': elapsed,
                        'output_tokens': response['usage']['completion_tokens'], **result})
                    print(args.engine, args.mode, language, task['id'], 'PASS' if result['passed'] else 'FAIL', flush=True)
                atomic_json(args.output/(language+'.json'), rows)
        summary = summarize(binding, tasks, reports)
        if args.smoke_limit:
            summary.update(completed=False, smoke_completed=True)
        atomic_json(summary_path, summary)
        atomic_json(args.output/'progress.json', {'status': 'completed', 'phase': 'saved',
            'completed': summary['total'], 'total': summary['total'], 'heartbeat_at': time.time()})
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('qwen3', 'qwen35'), required=True)
    parser.add_argument('--mode', choices=('base', 'adapter'), default='base')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, choices=(1, 2), default=2)
    parser.add_argument('--smoke-limit', type=int, choices=range(1, 11))
    parser.add_argument('--suite', choices=('audit', 'repairs'), default='audit')
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--repairs', type=Path)
    parser.add_argument('--replay-data', type=Path)
    args = parser.parse_args()
    if args.mode == 'adapter' and not args.adapter:
        parser.error('Adapter mode needs --adapter')
    if args.suite == 'repairs' and not all((args.selection, args.repairs, args.replay_data)):
        parser.error('Repair evaluation requires frozen selection, prompt data and replay data')
    signal.signal(signal.SIGTERM, lambda *_: exit(143))
    run(args)


if __name__ == '__main__':
    main()
