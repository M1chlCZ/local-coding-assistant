"""Execute authored reference policies through the real REPL; no model training or GPU use."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
from unittest.mock import patch

import recursive_agent as agent
from qwen35_train import atomic_json, load_verified, sha256

ROOT = Path(__file__).resolve().parent
TASKS = ROOT/'research/repair_tasks.json'
PUBLIC_INSTRUCTION = ' Run test_visible.py before and after the repair. Keep the test file unchanged.'
LIMITS = dict(depth=2, calls=8, output_tokens=6144, seconds=240)
INVALID_ACTION = '```python\nprint("This fence does not execute")\n```'


class TrajectoryOutputLimit(ValueError):
    def __init__(self, result, executions):
        super().__init__(result['error'])
        self.result, self.executions = result, executions


def training_tasks():
    tasks = [task for task in json.loads(TASKS.read_text()) if task['split'] == 'train']
    if len(tasks) != 20 or len({task['repository'] for task in tasks}) != 20:
        raise ValueError('Expected twenty distinct authored training families')
    return tasks


def task_variant(original, visible):
    if original['split'] != 'train' or 'test_visible.py' in original['editable']:
        raise ValueError('Lessons require training tasks with immutable public tests')
    task = copy.deepcopy({key: original[key] for key in
        ('id', 'repository', 'split', 'prompt', 'files', 'editable', 'checks')})
    if not visible:
        del task['files']['test_visible.py']
        task['prompt'] = task['prompt'].replace(PUBLIC_INSTRUCTION, '')
        task['prompt'] += ' No public test file is supplied; inspect the source and return the repair.'
    return task


def public_test(passed):
    program = "import sys;sys.path.insert(0,'/workspace');exec(open('/workspace/test_visible.py').read())"
    return ("import subprocess,sys\n"
        f"tested = subprocess.run([sys.executable, '-B', '-I', '-c', {program!r}], "
        "capture_output=True, text=True, timeout=45)\n"
        "print('Public test exit:', tested.returncode)\nprint(tested.stdout+tested.stderr)\n"
        + ("assert tested.returncode == 0 and 'VISIBLE_CHECKS_PASSED' in tested.stdout\n" if passed
           else "assert tested.returncode != 0\n"))


def actions(task, reference, recursive, recovery=False):
    plan = []
    def add(code, kind='root'):
        plan.append(dict(kind=kind, response='```repl\n'+code+'\n```'))
    if recovery:
        plan.append(dict(kind='root', invalid=True, response=INVALID_ACTION))
    add("print(context['task'][:1500])\nprint(list(context['files'])[:20])\n"
        "for name in context['editable'][:2]:\n    print(name, context['files'][name][:2000])")
    visible = 'test_visible.py' in task['files']
    if visible:
        add(public_test(False))
    if recursive:
        add("import json\nselected = {name: context['files'][name] for name in context['editable']}\n"
            "child_reply = rlm_query('Repair the selected source for this task. Return a JSON object "
            "mapping these paths to complete corrected source, through answer content. Task: ' "
            "+ context['task'] + '\\nSelected files: ' + json.dumps(selected))\nprint(child_reply)")
        add('print(context[:4000])', 'subcall')
        add("answer['content'] = " + repr(reference) + "\nanswer['ready'] = True", 'subcall')
        assignment = 'import json\npatch = json.loads(child_reply)\n'
    else:
        assignment = 'patch = ' + repr(reference) + '\n'
    add(assignment + "from pathlib import Path\nfor name,source in patch.items():\n"
        "    Path('/workspace', name).write_text(source)\n" + (public_test(True) if visible else '')
        + "print('Repair applied')")
    add("answer['content'] = patch\nanswer['ready'] = True")
    return plan


def trajectory(original, recursive, visible, revision, *, plan=None, instruction=''):
    from rlm.clients.openai import OpenAIClient
    from rlm.core.types import ModelUsageSummary
    if not re.fullmatch('[a-f0-9]{40}', revision):
        raise ValueError('Use the full authored-source Git revision')
    task = task_variant(original, visible)
    reference = agent.parse_patch(json.dumps(original['reference_patch']), task)
    if agent.grade(task, {name: task['files'][name] for name in task['editable']})['passed']:
        raise ValueError('The authored negative control must fail')
    if plan is None:
        plan = actions(task, reference, recursive, recovery=visible and not recursive)
    position, executions = 0, []
    execute = agent.Sandbox.execute_code

    def observe(environment, code):
        result = execute(environment, code)
        executions.append(dict(code=code, stdout=result.stdout, stderr=result.stderr))
        return result

    def policy(prompt, model=None):
        nonlocal position
        if position >= len(plan):
            raise ValueError('Reference policy requested an unexpected action')
        action = plan[position]
        if (agent.CALL_ORIGIN.get() or 'root') != action['kind']:
            raise ValueError('Reference policy call origin changed')
        position += 1
        return action['response']

    with patch.object(OpenAIClient, 'completion', side_effect=policy), \
         patch.object(OpenAIClient, 'get_last_usage', return_value=ModelUsageSummary(1, 1, 1)), \
         patch.object(agent.Sandbox, 'execute_code', observe):
        result = agent.solve(task, 'http://127.0.0.1:1', mode='rlm', instruction=instruction, **LIMITS)
    if result.get('error') == 'RuntimeError: Cell output limit exceeded' and not result.get('infrastructure_errors'):
        raise TrajectoryOutputLimit(result, executions)
    valid = [action for action in plan if not action.get('invalid')]
    codes = [action['response'][8:-4].strip() for action in valid]
    if (result.get('error') or result.get('infrastructure_errors')
            or result.get('patch') != reference or position != len(plan)
            or result['calls'] != len(plan) or result['calls'] > LIMITS['calls']
            or Counter(codes) != Counter(item['code'].strip() for item in executions)
            or any(item['stderr'] for item in executions)):
        raise ValueError('Reference trajectory failed execution: '+str(result.get('error', executions)))
    expected_hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in reference.items()}
    if result.get('workspace_hashes') != expected_hashes or not agent.grade(task, result['patch'])['passed']:
        raise ValueError('Applied reference patch failed independent grading')
    result['passed'] = True
    if recursive:
        queries = result.get('queries', [])
        if (len(queries) != 1 or not queries[0]['recursive']
                or agent.parse_patch(queries[0]['response'], task) != reference
                or queries[0]['execution_end']-queries[0]['execution_start'] != 2):
            raise ValueError('Reference child did not execute and return the applied patch')
    elif result.get('queries'):
        raise ValueError('Sequential reference unexpectedly delegated')
    if len(result['trace']) != len(plan):
        raise ValueError('Reference trace is incomplete')
    variant = ('recursive' if recursive else 'direct') + ('-public' if visible else '-source')
    provenance = dict(dataset='local/authored-agent-repairs', revision=revision,
        source_row_sha256=hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
        construction='scripted_reference_actions', model_discovered=False)
    rows = []
    for index, (call, action) in enumerate(zip(result['trace'], plan)):
        if call['response'] != action['response'] or call['kind'] != action['kind']:
            raise ValueError('Reference trace differs from the executed policy')
        if action.get('invalid'):
            continue
        messages = call['messages'] + [dict(role='assistant', content=call['response'])]
        if original['checks'] in '\n'.join(message['content'] for message in messages):
            raise ValueError('Private grading checks leaked into a lesson')
        rows.append(dict(task_id=original['id']+':'+variant+f':action-{index}',
            source_task_id=original['id'], repository=original['repository'], language='python',
            split='train', example_kind='executed_repl_repair', origin=call['kind'],
            variant=variant, messages=messages, provenance=provenance,
            verification=dict(kind='local_execution', passed=True)))
    return dict(task_id=original['id'], variant=variant, rows=rows, result=result,
                executions=executions, verification=dict(negative_control_failed=True, final_patch_passed=True),
                usage_note='Scripted reference policy; usage counters are placeholders, not model token measurements.')


def anchors(path):
    rows, manifest = load_verified(path, Path(str(path)+'.manifest.json'))
    if manifest['verification'] != 'local_execution':
        raise ValueError('Anchors require locally verified training rows')
    selected, counts = [], Counter()
    for row in rows:
        if counts[row['language']] < 30 and row.get('example_kind', 'complete_code') in ('complete_code', 'teacher_reference'):
            selected.append(row)
            counts[row['language']] += 1
    return selected


def write_dataset(output, rows, binding):
    output = Path(output)
    dataset = output/'train.jsonl'
    dataset.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    metadata = dict(schema_version=1, split='train', verification='local_execution',
        rows=len(rows), languages=dict(Counter(row['language'] for row in rows)),
        dataset_sha256=sha256(dataset), binding=binding,
        scope='Executed reference policies plus verified training rows; not evidence of model-discovered solutions.')
    atomic_json(Path(str(dataset)+'.manifest.json'), metadata)
    load_verified(dataset, Path(str(dataset)+'.manifest.json'))
    return metadata


def verify_tokens(output, revision, tokenizer=None):
    """Measure all executed responses, including mistakes masked out of training."""
    from qwen35_train import MODEL, encode_rows, validate_tokenizer
    output = Path(output)
    dataset = output/'train.jsonl'
    manifest = Path(str(dataset)+'.manifest.json')
    rows, metadata = load_verified(dataset, manifest)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision,
            local_files_only=True, trust_remote_code=False)
        validate_tokenizer(tokenizer)
    _, encoded = encode_rows(tokenizer, rows, 4096)
    records = [json.loads(path.read_text()) for path in sorted((output/'traces').glob('*.json'))]
    expected = [row for record in records for row in record['rows']]
    actual = [row for row in rows if row.get('example_kind') == 'executed_repl_repair']
    complete = bool(records) and sorted(expected, key=lambda r:r['task_id']) == sorted(actual, key=lambda r:r['task_id'])
    actions, totals = [], []
    for record in records:
        trace = record['result']['trace']
        complete = complete and bool(trace) and len(trace) == record['result']['calls']
        exported = {row['task_id']: row for row in record['rows']}
        matched = set()
        for index, call in enumerate(trace):
            identity = record['task_id']+':'+record['variant']+f':action-{index}'
            row = exported.get(identity)
            if row is not None:
                matched.add(identity)
                complete = complete and row['messages'] == call['messages']+[
                    dict(role='assistant',content=call['response'])] and row['origin'] == call['kind']
            else:
                complete = complete and (index == 0 and record['variant'] == 'direct-public'
                    and call['kind'] == 'root' and call['response'] == INVALID_ACTION)
        complete = complete and matched == set(exported)
        measured = [dict(task_id=str(i), language='python', example_kind='executed_repl_repair',
            messages=call['messages']+[dict(role='assistant',content=call['response'])])
            for i,call in enumerate(trace)]
        _, report = encode_rows(tokenizer, measured, 4096)
        actions.extend(report['included']+report['excluded'])
        total = sum(row['answer_tokens'] for row in report['included']+report['excluded'])
        totals.append(total)
        complete = complete and not report['excluded_count'] and len(trace) <= LIMITS['calls'] and total <= LIMITS['output_tokens']
    binding = metadata['binding']
    unchanged = all(sha256(ROOT/name)==digest for name,digest in binding['sources'].items())
    result = dict(passed=bool(complete and unchanged and not encoded['excluded_count']),
        dataset_sha256=sha256(dataset), manifest_sha256=sha256(manifest),
        tokenizer_model=MODEL, tokenizer_revision=revision, sources_unchanged=unchanged,
        all_agent_actions_included=not encoded['excluded_count'], complete_trajectories=bool(complete),
        included_count=encoded['included_count'], excluded_count=encoded['excluded_count'],
        max_action_tokens=max((a['answer_tokens'] for a in actions),default=0),
        max_trajectory_output_tokens=max(totals,default=0),
        max_context_tokens=max((a['tokens'] for a in actions),default=0))
    atomic_json(output/'feasibility.json', result)
    if not result['passed']:
        raise ValueError('Complete trajectories do not fit the runtime budgets; see feasibility.json')
    return result


def build(output, revision, anchor_path=None):
    if not re.fullmatch('[a-f0-9]{40}', revision):
        raise ValueError('Use the full authored-source Git revision')
    anchor_rows = anchors(anchor_path) if anchor_path else []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'traces').mkdir()
    sources = ('agent_curriculum.py', 'recursive_agent.py', 'rlm_worker.py', 'requirements-rlm.txt',
               'evaluation.py', 'qwen35_train.py', 'research/repair_tasks.json')
    binding = dict(revision=revision, upstream=agent.UPSTREAM, limits=LIMITS,
        sources={name: sha256(ROOT/name) for name in sources},
        anchors_sha256=sha256(anchor_path) if anchor_path else None)
    atomic_json(output/'binding.json', binding)
    rows = []
    for task in training_tasks():
        for recursive in (False, True):
            for visible in (True, False):
                record = trajectory(task, recursive, visible, revision)
                atomic_json(output/'traces'/(task['id']+'-'+record['variant']+'.json'), record)
                rows.extend(record['rows'])
                print(f"Verified {task['id']} {record['variant']}: {len(record['rows'])} actions", flush=True)
    rows.extend(anchor_rows)
    metadata = write_dataset(output, rows, binding)
    subprocess.run([str(ROOT/'.cache/qwen35-env/bin/python'), str(ROOT/'agent_curriculum.py'),
                    '--output', str(output), '--verify-tokens'], check=True)
    print(json.dumps(dict(rows=metadata['rows'], languages=metadata['languages'], output=str(output))), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', help='Full authored-source Git revision')
    parser.add_argument('--anchors', type=Path, help='Locally verified JSONL with adjacent .manifest.json')
    parser.add_argument('--verify-tokens', action='store_true', help='Check an existing collection with the pinned tokenizer')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.verify_tokens:
        from coding_loop_stage import REVISION
        print(json.dumps(verify_tokens(args.output, REVISION)))
    else:
        if not args.revision: parser.error('--revision is required for collection')
        build(args.output, args.revision, args.anchors)


if __name__ == '__main__': main()
