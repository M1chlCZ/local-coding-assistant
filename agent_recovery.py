"""Finite corrective imitation: execute student prefixes, then verified train-only corrections."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from unittest.mock import patch

import recursive_agent as agent
from agent_curriculum import LIMITS, anchors, public_test, task_variant, training_tasks, write_dataset
from agent_pilot import RECURSIVE, recursive_success
from qwen35_train import atomic_json, load_verified, sha256

ROOT = Path(__file__).resolve().parent


def actions(task, reference, recursive):
    """Leave room for student mistakes; applying, testing and finishing is one action."""
    plan = []
    def add(code, kind='root'):
        plan.append(dict(kind=kind, response='```repl\n'+code+'\n```'))
    add("print(context['task'])\n"
        "for name in context['editable']:\n    print(name, context['files'][name])\n"
        "print(context['files'].get('test_visible.py', 'No public tests'))\n"
        + public_test(False).replace('assert tested.returncode != 0\n', ''))
    if recursive:
        add("import json\nselected = {p: context['files'][p] for p in context['editable']}\n"
            "child_reply = rlm_query('Return a JSON patch implementing every requirement, including "
            "edge cases. Task: ' + context['task'] + '\\nSource: ' + json.dumps(selected))\n"
            "print(child_reply)")
        add('print(context[:6000])', 'subcall')
        add("answer['content'] = " + repr(reference) + "\nanswer['ready'] = True", 'subcall')
        assignment = 'import json\npatch = json.loads(child_reply)\n'
    else:
        assignment = 'patch = ' + repr(reference) + '\n'
    add(assignment + "from pathlib import Path\n"
        "assert patch and set(patch).issubset(context['editable'])\n"
        "for name, source in patch.items():\n    Path('/workspace', name).write_text(source)\n"
        + public_test(True) + "answer['content'] = patch\nanswer['ready'] = True")
    return plan


def trajectory(original, recursive, prefix, revision, base='http://127.0.0.1:8090'):
    from rlm.clients.openai import OpenAIClient
    from rlm.core.types import ModelUsageSummary
    if original['split'] != 'train' or prefix not in (0, 2) or not re.fullmatch('[a-f0-9]{40}', revision):
        raise ValueError('Only pinned training families and declared student prefixes are allowed')
    task = task_variant(original, True)
    reference = agent.parse_patch(json.dumps(original['reference_patch']), task)
    variant = ('recursive' if recursive else 'ordinary') + f'-prefix-{prefix}'
    record = dict(task_id=task['id'], variant=variant, prefix=prefix, recursive=recursive,
                  accepted=False, rows=[], teacher_calls=[])
    if (agent.grade(task, {p: task['files'][p] for p in task['editable']})['passed']
            or not agent.grade(task, reference)['passed']):
        return dict(record, reason='Invalid negative control or reference repair')
    plan = actions(task, reference, recursive)
    root_calls = position = calls = 0
    teaching = False
    completion, usage = OpenAIClient.completion, OpenAIClient.get_last_usage

    def policy(client, prompt, model=None):
        nonlocal root_calls, position, calls, teaching
        index = calls
        calls += 1
        origin = agent.CALL_ORIGIN.get() or 'root'
        if origin == 'root' and root_calls >= prefix:
            teaching = True
        if not teaching:
            root_calls += origin == 'root'
            response = completion(client, prompt, model)
            client._recovery_usage = usage(client)
            return response
        if position >= len(plan) or origin != plan[position]['kind']:
            raise ValueError('Correction did not follow its declared executable continuation')
        response = plan[position]['response']
        position += 1
        record['teacher_calls'].append(index)
        # Real token counts are measured offline before training, including masked student actions.
        client._recovery_usage = ModelUsageSummary(1, 1, 1)
        return response

    with patch.object(OpenAIClient, 'completion', policy), \
         patch.object(OpenAIClient, 'get_last_usage', lambda client: client._recovery_usage):
        result = agent.solve(task, base, instruction=RECURSIVE if recursive else '', **LIMITS)
    record['result'] = result
    if result.get('patch'):
        result.update(agent.grade(task, result['patch']))
    expected = {p: hashlib.sha256(s.encode()).hexdigest() for p, s in reference.items()}
    if (result.get('error') or result.get('infrastructure_errors') or not result['passed']
            or result.get('patch') != reference or result.get('workspace_hashes') != expected
            or position != len(plan) or result['calls'] > LIMITS['calls']
            or recursive and not recursive_success(result, task)):
        return dict(record, reason='Unverified or incomplete correction; no training rows')
    provenance = dict(dataset='local/student-state-agent-repairs', revision=revision,
        source_row_sha256=hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
        construction='executed_expert_correction_after_base_policy_prefix', model_discovered=False)
    for index in record['teacher_calls']:
        call = result['trace'][index]
        messages = call['messages'] + [dict(role='assistant', content=call['response'])]
        if original['checks'] in '\n'.join(m['content'] for m in messages):
            raise ValueError('Private checks leaked into correction messages')
        record['rows'].append(dict(task_id=task['id']+':'+variant+f':action-{index}',
            source_task_id=task['id'], repository=original['repository'], language='python', split='train',
            example_kind='executed_repl_repair', origin=call['kind'], variant=variant,
            messages=messages, provenance=provenance,
            verification=dict(kind='local_execution', passed=True)))
    return dict(record, accepted=True)


def finalize(folder, revision, anchor_path, tokenizer=None):
    from qwen35_train import MODEL, encode_rows, validate_tokenizer
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision,
            local_files_only=True, trust_remote_code=False)
        validate_tokenizer(tokenizer)
    rows, records, excluded, measured = [], [], [], []
    files = sorted((folder/'traces').glob('*.json'))
    if len(files) != len(training_tasks()) * 4:
        raise ValueError('Collection must finish every declared training trajectory')
    for path in files:
        record = json.loads(path.read_text())
        if not record['accepted']:
            excluded.append(dict(file=path.name, reason=record['reason']))
            continue
        trace = record['result']['trace']
        all_actions = [dict(task_id=str(i), language='python', example_kind='executed_repl_repair',
            messages=c['messages']+[dict(role='assistant', content=c['response'])]) for i,c in enumerate(trace)]
        _, counts = encode_rows(tokenizer, all_actions, 4096)
        wanted = [all_actions[i]['messages'] for i in record['teacher_calls']]
        if (wanted != [r['messages'] for r in record['rows']]
                or len(trace) != record['result']['calls'] or len(trace) > LIMITS['calls']):
            raise ValueError('Correction targets no longer match the executed trace')
        if counts['excluded_count'] or sum(a['answer_tokens'] for a in counts['included']) > LIMITS['output_tokens']:
            excluded.append(dict(file=path.name, reason='Whole trajectory exceeds context or action budget'))
            continue
        rows.extend(record['rows']); records.append(record); measured.extend(counts['included'])
    recovered = {r['task_id'] for r in records if r['prefix'] == 2}
    if len(recovered) < 16 or not all(any(r['prefix']==2 and r['recursive']==mode for r in records) for mode in (False,True)):
        raise ValueError('Insufficient actual student-state coverage for this declared recipe')
    binding = json.loads((folder/'binding.json').read_text())
    if any(sha256(ROOT/name) != digest for name,digest in binding['sources'].items()):
        raise ValueError('Collection sources changed')
    rows.extend(anchors(anchor_path))
    metadata = write_dataset(folder, rows, dict(binding, traces={p.name:sha256(p) for p in files},
        construction='student_state_corrective_imitation', excluded_trajectories=excluded))
    _, encoded = encode_rows(tokenizer, rows, 4096)
    if encoded['excluded_count']:
        raise ValueError('A selected correction or retention anchor does not fit')
    dataset = folder/'train.jsonl'
    evidence = dict(completed=True, passed=True, all_agent_actions_included=True, complete_trajectories=True,
        dataset_sha256=sha256(dataset), manifest_sha256=sha256(str(dataset)+'.manifest.json'),
        tokenizer_model=MODEL, tokenizer_revision=revision, included_count=len(rows), excluded_count=0,
        max_action_tokens=max(a['answer_tokens'] for a in measured),
        max_context_tokens=max(a['tokens'] for a in measured),
        accepted_trajectories=len(records), excluded_trajectories=excluded,
        recovered_training_families=len(recovered), sources_unchanged=True)
    atomic_json(folder/'feasibility.json', evidence)
    return evidence


def collect(folder, revision, anchor_path):
    from coding_loop_stage import server, REVISION
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'traces').mkdir(exist_ok=True)
    sources = ('agent_recovery.py','agent_curriculum.py','agent_pilot.py','recursive_agent.py',
               'rlm_worker.py','qwen35_server.py','qwen35_train.py','research/repair_tasks.json')
    binding = dict(revision=revision, policy_model_revision=REVISION, policy_adapter=None,
        sources={n:sha256(ROOT/n) for n in sources}, anchors_sha256=sha256(anchor_path), limits=LIMITS)
    path = folder/'binding.json'
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError('Cannot resume collection with changed sources or policy')
    atomic_json(path, binding)
    with server(None, folder):
        for task in training_tasks():
            for recursive in (False, True):
                for prefix in (0, 2):
                    name = task['id']+('-recursive' if recursive else '-ordinary')+f'-prefix-{prefix}.json'
                    target = folder/'traces'/name
                    if target.exists(): continue
                    record = trajectory(task, recursive, prefix, revision)
                    atomic_json(target, record)
                    print(name, 'accepted='+str(record['accepted']), flush=True)
    subprocess.run([str(ROOT/'.cache/qwen35-env/bin/python'),str(ROOT/'agent_recovery.py'),
        'finalize','--folder',str(folder),'--revision',REVISION,'--anchors',str(anchor_path)],check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('collect','finalize'))
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--anchors', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'collect': collect(args.folder, args.revision, args.anchors)
    else: print(json.dumps(finalize(args.folder,args.revision,args.anchors)))


if __name__ == '__main__': main()
