"""Finite retained-BASE interface ablation; it cannot authorize training or Collect."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from agent_pilot import recursive_success, tested_repl
from coding_loop_stage import REVISION, progress
from evaluation import request
from qwen35_train import MODEL, atomic_json, sha256
from recursive_agent import grade, parse_patch, solve as fenced_solve
from repl_learning import run_child
from research.tool_protocol import solve

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT/'research/tool_protocol_tasks.json'
MODES = ('fenced', 'typed', 'constrained')
LIMITS = dict(calls=12, output_tokens=8192, seconds=360)
BASE = 'http://127.0.0.1:8090'
SOURCES = ('research/tool_protocol.py', 'research/tool_protocol_compare.py',
    'research/tool_protocol_tasks.json', 'recursive_agent.py', 'rlm_worker.py', 'repl_actions.py',
    'agent_pilot.py', 'evaluation.py', 'qwen35_server.py', 'qwen35_train.py', 'structured_output.py',
    'research/adapter_eval_server.py', 'repl_learning.py', 'coding_loop_stage.py',
    'focused_experiment.py', 'requirements-structured.txt', 'requirements-rlm.txt')


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    return {name: sha256(ROOT/name) for name in SOURCES}


def summarize(rows, entries):
    expected = {(e['task']['id'], m) for e in entries for m in MODES}
    indexed = {(r['id'], r['interface']): r for r in rows}
    if set(indexed) != expected or len(indexed) != len(rows):
        raise ValueError('Complete unique matched results required')
    summary = dict(completed=True, tasks_per_interface=len(entries), learning_gain_established=False,
        full_loop_authorized=False, independent_manual_review_pending=True, interfaces={}, versus_fenced={},
        scope='Same retained BASE, six public development canaries, no training or reserved heldouts. '
              'Typed/constrained prompts match; fenced also changes the interface and prompt.')
    for mode in MODES:
        selected = [indexed[e['task']['id'], mode] for e in entries]
        summary['interfaces'][mode] = {key: sum(r[key] for r in selected) for key in
            ('passed', 'tested_repl', 'recursive_passed', 'calls', 'output_tokens', 'elapsed_s')}
        summary['interfaces'][mode]['protocol_errors'] = (None if mode == 'fenced' else
            sum(len(r.get('protocol_errors', [])) for r in selected))
        summary['interfaces'][mode]['ordinary_tested_repl'] = sum(
            indexed[e['task']['id'], mode]['tested_repl'] for e in entries if e['condition'] == 'ordinary')
        if mode != 'fenced':
            summary['versus_fenced'][mode] = {
                metric+'_'+kind: [e['task']['id'] for e in entries
                    if bool(indexed[e['task']['id'], 'fenced'][key]) == before
                    and bool(indexed[e['task']['id'], mode][key]) != before]
                for metric, key in [('functional', 'passed'), ('tested_repl', 'tested_repl'),
                                    ('recursive', 'recursive_passed')]
                for kind, before in [('gains', False), ('losses', True)]}
    return summary


def controls(entries):
    results = []
    for entry in entries:
        task = entry['task']
        assert len(task['editable']) == 2 and set(task['files']) == set(task['editable'])
        original = {p: task['files'][p] for p in task['editable']}
        assert not grade(task, original)['passed'], task['id']+' accepts no-op'
        assert grade(task, task['reference_patch'])['passed'], task['id']+' reference fails'
        for path in task['editable']:
            partial = dict(task['reference_patch'])
            partial[path] = task['files'][path]
            assert not grade(task, partial)['passed'], task['id']+' accepts missing edit '+path
        results.append(dict(id=task['id'], no_op_rejected=True, reference_passed=True, missing_edits_rejected=2))
    return dict(passed=True, tasks=results)


def evaluate(entry, mode):
    task = entry['task']
    recursive = entry['condition'] == 'recursive'
    if mode == 'fenced':
        result = fenced_solve(task, BASE, mode='rlm' if recursive else 'baseline',
            depth=2 if recursive else 1, action_api=True, context_window=True, **LIMITS,
            instruction='No test files or test functions are supplied. Derive actual assertions from '
            'the requirements. Reproduce a failure before editing, then test the repaired source.\n')
    else:
        result = solve(task, BASE, recursive=recursive, constrained=mode == 'constrained', **LIMITS)
    if result.get('patch'):
        result['grading'] = grade(task, result['patch'])
        result['passed'] = result['grading']['passed']
    result.update(tested_repl=tested_repl(result),
        recursive_passed=recursive_success(result, task) if recursive else False,
        interface=mode, condition=entry['condition'])
    return result


@contextmanager
def model_server(python, folder):
    # Isolated optional dependencies; never install into a shared training environment.
    command = [str(python), str(ROOT/'qwen35_server.py'), '--revision', REVISION,
        '--batch-size', '1', '--output-limit', '1024', '--context-length', '8192', '--structured-output']
    with (folder/'server.log').open('ab') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic()+900
        while True:
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Model failed to load; inspect server.log')
            try:
                health = request(BASE, '/health', timeout=2)
            except (OSError, ValueError):
                time.sleep(2)
                continue
            if (health['revision'] != REVISION or health['adapter_available']
                    or health['adapter'] or not health['structured_outputs']):
                raise RuntimeError('Unexpected model binding')
            break
        yield health
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def compare(folder, python, gpu_lock):
    target = folder/'comparison'
    if target.exists():
        raise FileExistsError('Existing partial or complete evidence must not be overwritten')
    with Path(gpu_lock).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for port in (8080, 8090):
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', port)) == 0:
                    raise RuntimeError('An existing model server is running')
        target.mkdir()
        entries = read(TASKS)
        atomic_json(target/'controls.json', controls(entries))
        binding = dict(model=MODEL, revision=REVISION, adapter=None, precision='bf16', sources=sources(),
            limits=LIMITS, context=8192, action_tokens=1024, temperature=0, batch_size=1,
            order=[(e['task']['id'], MODES[i % 3:]+MODES[:i % 3]) for i,e in enumerate(entries)])
        from rlm.core import rlm
        from rlm.utils import parsing
        binding['upstream'] = dict(rlm=sha256(rlm.__file__), parsing=sha256(parsing.__file__))
        runtime = subprocess.check_output([str(python), '-c',
            'import json,importlib.metadata as m;print(json.dumps({p:m.version(p) for p in '
            '("torch","transformers","unsloth","xgrammar","jsonschema")}))'], text=True)
        binding['runtime'] = json.loads(runtime)
        atomic_json(target/'binding.json', binding)
        rows = []
        with model_server(python, target) as health:
            atomic_json(target/'health.json', health)
            for index, entry in enumerate(entries):
                for mode in MODES[index % 3:]+MODES[:index % 3]:
                    progress(folder, 'compare-tools', len(rows), len(entries)*3,
                             task=entry['task']['id'], interface=mode)
                    atomic_json(target/'task.json', dict(entry=entry, interface=mode))
                    output = target/'task-result.json'
                    output.unlink(missing_ok=True)
                    run_child([sys.executable, '-m',
                        'research.tool_protocol_compare', 'task', '--folder', str(folder)],
                        LIMITS['seconds']+180)
                    row = read(output)
                    if (row['id'], row['interface']) != (entry['task']['id'], mode):
                        raise ValueError('Unexpected task result')
                    rows.append(row)
                    atomic_json(target/'results.json', dict(completed=False, rows=rows))
                    print(mode, row['id'], 'functional', row['passed'],
                          'tested', row['tested_repl'], 'recursive', row['recursive_passed'], flush=True)
                    if request(BASE, '/health', timeout=180) != health:
                        raise RuntimeError('Model health/binding changed')
        if sources() != binding['sources']:
            raise RuntimeError('Comparison sources changed')
        atomic_json(target/'results.json', dict(completed=True, rows=rows))
        atomic_json(target/'summary.json', summarize(rows, entries))
        progress(folder, 'compare-tools', len(rows), len(rows))


def verify(folder):
    """CPU verification replays token counts and independently regrades all returned patches."""
    target = folder/'comparison'
    binding, saved = read(target/'binding.json'), read(target/'results.json')
    if not saved['completed'] or binding['sources'] != sources():
        raise ValueError('Incomplete or changed evidence')
    entries = read(TASKS)
    tasks = {e['task']['id']: e['task'] for e in entries}
    summary = summarize(saved['rows'], entries)
    if summary != read(target/'summary.json'):
        raise ValueError('Summary does not match raw results')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                                               trust_remote_code=False)
    roots = children = generations = 0
    regrades = []
    initial = {}
    for row in saved['rows']:
        task = tasks[row['id']]
        if row.get('infrastructure_errors'):
            raise ValueError('Infrastructure failure cannot be scored silently')
        if row.get('patch'):
            roots += 1
            checked = grade(task, row['patch'])
            assert checked['passed'] == row['passed']
            regrades.append(dict(id=row['id'], interface=row['interface'], origin='root',
                patch_sha256=hashlib.sha256(json.dumps(row['patch'], sort_keys=True).encode()).hexdigest(),
                grading=checked))
        else:
            assert not row['passed']
        for index, query in enumerate(row['queries']):
            try:
                proposal = parse_patch(query['response'], task)
            except (ValueError, TypeError):
                regrades.append(dict(id=row['id'], interface=row['interface'], origin='child',
                                     query_index=index, invalid_patch=True))
                continue
            children += 1
            regrades.append(dict(id=row['id'], interface=row['interface'], origin='child', query_index=index,
                patch_sha256=hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest(),
                grading=grade(task, proposal)))
        assert tested_repl(row) == row['tested_repl']
        assert (recursive_success(row, task) if row['condition'] == 'recursive' else False) == row['recursive_passed']
        assert len(row['trace']) == len(row['context_windows'])
        for call, receipt in zip(row['trace'], row['context_windows']):
            encoded = tokenizer.apply_chat_template(call['messages'], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=True)['input_ids']
            assert len(encoded) == call['input_tokens'] == receipt['prompt_tokens']
            assert len(encoded)+receipt['output_tokens'] <= 8192
            from qwen35_server import prepare_context
            replay = prepare_context(tokenizer, dict(messages=receipt['requested_messages'],
                max_completion_tokens=receipt['output_tokens']), 8192, 1024)
            assert all(receipt[k] == v for k,v in replay.items())
            generations += 1
            if row['interface'] != 'fenced':
                from research.tool_protocol import action_code
                try:
                    action_code(call['raw_response'], row['condition'] != 'recursive' or call['kind'] == 'subcall')
                except ValueError:
                    pass  # Invalid actions remain model failures, never repaired in verification.
        if row['trace'] and row['interface'] != 'fenced':
            initial[row['id'], row['interface']] = row['trace'][0]['messages']
    assert all(initial[e['task']['id'], 'typed'] == initial[e['task']['id'], 'constrained'] for e in entries)
    atomic_json(folder/'regrades.json', dict(completed=True, rows=regrades))
    atomic_json(folder/'verified.json', dict(completed=True, passed=True, root_patches_regraded=roots,
        child_patches_regraded=children, generations_replayed=generations,
        results_sha256=sha256(target/'results.json'), summary_sha256=sha256(target/'summary.json'),
        binding_sha256=sha256(target/'binding.json'), manual_review_required=True,
        regrades_sha256=sha256(folder/'regrades.json'),
        learning_gain_established=False, full_loop_authorized=False))


def main():
    def interrupt(*unused):
        raise KeyboardInterrupt('Finite protocol comparison interrupted')
    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['compare', 'task', 'verify'])
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--server-python', type=Path)
    parser.add_argument('--gpu-lock', type=Path)
    args = parser.parse_args()
    if args.phase == 'task':
        job = read(args.folder/'comparison/task.json')
        atomic_json(args.folder/'comparison/task-result.json', evaluate(job['entry'], job['interface']))
    elif args.phase == 'verify':
        verify(args.folder)
    else:
        if not args.server_python or not args.gpu_lock:
            parser.error('compare requires --server-python and --gpu-lock')
        compare(args.folder, args.server_python, args.gpu_lock)


if __name__ == '__main__':
    main()
