"""Replay recorded train-only prefixes and execute verified corrective continuations.

No model generation or optimization occurs. Prefixes remain conditioning history.
"""
import copy
import hashlib
import io
import json
import re
from unittest.mock import patch

from agent_atomic import LIMITS
from agent_pilot import recursive_success
from qwen35_server import prepare_context
import recursive_agent as agent


def response_tokens(tokenizer, response):
    # Scripted completions terminate with one EOS, as the native evaluator does.
    return len(tokenizer.encode(response, add_special_tokens=False)) + 1


def verify_result(task, recursive, result):
    reference = task['reference_patch']
    broken = {p: task['files'][p] for p in task['editable']}
    if agent.grade(task, broken)['passed'] or not agent.grade(task, reference)['passed']:
        raise ValueError('Invalid broken control or reference repair')
    if (result.get('error') or result.get('infrastructure_errors')
            or result.get('patch') != reference or not agent.grade(task, result['patch'])['passed']):
        raise ValueError('Correction did not return the verified reference repair')
    result['passed'] = True
    if not agent.tested_repl(result):
        raise ValueError('Final returned source was not checked in the parent')
    if recursive:
        if not recursive_success(result, task):
            raise ValueError('No genuinely checked child repair applied by the parent')
    elif result.get('queries'):
        raise ValueError('Ordinary correction unexpectedly delegated')
    checks = [c for e in result['executions'] for c in e.get('checks', [])]
    if not any(c['assertions'] and c['returncode'] != 0 for c in checks):
        raise ValueError('Correction lacks a discriminating failed-before check')


def replay(original, recursive, source, prefix, continuation, revision, tokenizer):
    """Reject any changed requested/effective prefix; label only new verified actions."""
    from rlm.clients.openai import OpenAIClient
    from rlm.core.types import ModelUsageSummary
    if (original.get('split') != 'train' or not re.fullmatch('[a-f0-9]{40}', revision)
            or type(prefix) is not int or not 0 < prefix <= len(source['trace'])
            or len(source.get('context_windows', [])) != len(source['trace'])
            or not continuation or prefix + len(continuation) > LIMITS['calls']):
        raise ValueError('Only complete bounded training-prefix corrections are allowed')
    task = copy.deepcopy({k: original[k] for k in
        ('id', 'repository', 'split', 'prompt', 'files', 'editable', 'checks')})
    plan = [dict(c, train=False) for c in source['trace'][:prefix]] + continuation
    if any(sum(c['kind'] == kind for c in plan) > 8 for kind in ('root', 'subcall')):
        raise ValueError('Complete correction exceeds the unchanged iteration budget')
    position = 0

    def context(request, timeout):
        if request.full_url != 'http://127.0.0.1:1/context':
            raise ValueError('Replay permits CPU context preparation only')
        data = json.loads(request.data)
        prepared = prepare_context(tokenizer, data, 8192, 1024)
        if position <= prefix and position < len(source['trace']):
            recorded = source['context_windows'][position]
            if data['messages'] != recorded['requested_messages']:
                raise ValueError(f'Recorded requested prefix differs at call {position}')
            if prepared != {k:v for k,v in recorded.items() if k not in ('requested_messages','kind')}:
                raise ValueError(f'Recorded effective context differs at call {position}')
        return io.BytesIO(json.dumps(prepared).encode())

    def policy(client, prompt, model=None):
        nonlocal position
        if position >= len(plan):
            raise ValueError('Unexpected completion beyond the declared correction')
        action = plan[position]
        if (agent.CALL_ORIGIN.get() or 'root') != action['kind']:
            raise ValueError('Recorded or corrective call origin changed')
        inputs = len(tokenizer.apply_chat_template(prompt, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=True)['input_ids'])
        outputs = response_tokens(tokenizer, action['response'])
        if position < prefix:
            if prompt != action['messages'] or inputs != action['input_tokens']:
                raise ValueError(f'Recorded effective prefix differs at call {position}')
            # Historical counters are retained; absent token IDs prevent reconstructing EOS exactly.
            if action['output_tokens'] not in (outputs - 1, outputs):
                raise ValueError(f'Recorded output count cannot be reproduced at call {position}')
            outputs = action['output_tokens']
        if outputs > client.sampling_args['max_completion_tokens']:
            raise ValueError('Complete response exceeds the remaining output budget')
        client._correction_usage = ModelUsageSummary(1, inputs, outputs)
        position += 1
        return action['response']

    with patch.object(agent, 'urlopen', context), \
         patch.object(OpenAIClient, 'completion', policy), \
         patch.object(OpenAIClient, 'get_last_usage', lambda client: client._correction_usage):
        result = agent.solve(task, 'http://127.0.0.1:1', mode='rlm' if recursive else 'baseline',
            action_api=True, context_window=True, **LIMITS)
    if position != len(plan) or len(result['trace']) != len(plan):
        raise ValueError('Correction stopped before its complete declared continuation')
    verify_result(original, recursive, result)
    variant = ('recursive' if recursive else 'ordinary') + f'-prefix-{prefix}'
    provenance = dict(dataset='local/recorded-policy-corrections', revision=revision,
        source_row_sha256=hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
        prefix_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        construction='reexecuted_recorded_prefix_with_scripted_correction', model_discovered=False)
    targets, rows = [], []
    for i, (call, action) in enumerate(zip(result['trace'], plan)):
        if call['response'] != action['response'] or call['kind'] != action['kind']:
            raise ValueError('Executed trace differs from declared correction')
        if i < prefix or action.get('train') is False:
            continue
        messages = call['messages'] + [dict(role='assistant', content=call['response'])]
        if original['checks'] in '\n'.join(m['content'] for m in messages):
            raise ValueError('Private grading checks leaked into correction messages')
        targets.append(i)
        rows.append(dict(task_id=task['id']+':'+variant+f':action-{i}',
            source_task_id=task['id'], repository=task['repository'], language='python', split='train',
            example_kind='executed_repl_repair', origin=call['kind'], variant=variant,
            messages=messages, provenance=provenance, verification=dict(kind='local_execution',passed=True)))
    return dict(task_id=task['id'], variant=variant, recursive=recursive, prefix=prefix,
        prefix_reused=True, accepted=True, target_indices=targets, rows=rows, result=result,
        usage_note='Pinned CPU context policy; measured scripted tokens plus EOS. '
            'Historical output counters retained and bounded against canonical text tokenization; '
            'original generated token IDs were not recorded.')
