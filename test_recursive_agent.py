"""Runnable checks for the recursive agent's trust boundaries and real sandbox."""
import json
import time
import threading
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch as mock_patch

import recursive_agent as agent


def rejects(fn):
    try:
        fn()
    except (ValueError, RuntimeError, TimeoutError):
        return
    raise AssertionError('Expected rejection')


task = {'files': {'solver.py': 'x=1\n'}, 'editable': ['solver.py']}
assert agent.parse_patch('{"solver.py":"x=2\\n"}', task) == {'solver.py': 'x=2\n'}
for patch in ['{"../secret.py":"x=2"}', '{"checks.py":"pass"}', '[]', '{"solver.py":5}']:
    rejects(lambda: agent.parse_patch(patch, task))
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root/'solver.py').write_text('x=1')
    (root/'.private.py').write_text('secret')
    (root/'linked.py').symlink_to(root/'.private.py')
    assert agent.snapshot(root) == {'solver.py':'x=1'}
    (root/'large.py').write_text('x'*100001)
    rejects(lambda: agent.snapshot(root))
budget = agent.Budget(calls=1, output_tokens=10, seconds=1)
assert budget.reserve(20) == 10
rejects(lambda: budget.reserve(1))
budget = agent.Budget(calls=2, output_tokens=10, seconds=.01)
time.sleep(.02)
rejects(lambda: budget.reserve(1))

fixture = json.loads(agent.TASKS.read_text())[0]
for exit_call in ('raise SystemExit(0)', 'import os; os._exit(0)'):
    forged = {fixture['editable'][0]: 'print("RLM_CHECKS_PASSED")\n' + exit_call + '\n'}
    assert not agent.grade(fixture, forged)['passed'], 'Early exit must not forge a grading reward'
assert agent.grade(fixture, fixture['reference_patch'])['passed']

env = agent.Sandbox(context_payload={'files': {'solver.py': 'x=1'}}, cell_seconds=1)
try:
    import subprocess
    config = json.loads(subprocess.check_output(['docker','inspect',env.container]))[0]
    assert config['HostConfig']['NetworkMode'] == 'none'
    assert config['HostConfig']['ReadonlyRootfs'] and config['Config']['User'] == '65534:65534'
    assert not config['Mounts'] and 'ALL' in config['HostConfig']['CapDrop']
    first = env.execute_code('value = 41; print(context["files"]["solver.py"])')
    assert 'x=1' in first.stdout
    assert env.execute_code('print(value+1)').stdout.strip() == '42'
    assert env.execute_code('import solver; print(solver.x)').stdout.strip() == '1'
    rejects(lambda: env.execute_code('print("x"*200000)'))
finally:
    env.cleanup()
env = agent.Sandbox(context_payload={}, cell_seconds=.3)
try:
    rejects(lambda: env.execute_code('while True: pass'))
finally:
    env.cleanup()
env = agent.Sandbox(context_payload={}, cell_seconds=.3,
                    subcall_fn=lambda prompt: SimpleNamespace(response='x' * 8000))
started = time.monotonic()
try:
    rejects(lambda: env.execute_code('import os,json,time\n'
        'frame=(json.dumps({"type":"query","prompt":"x","recursive":True})+"\\n").encode()\n'
        'for _ in range(1000): os.write(1,frame)\ntime.sleep(10)'))
finally:
    env.cleanup()
assert time.monotonic() - started < 5, 'Blocked IPC write must be bounded'
assert not env.reader.is_alive() and not env.writer.is_alive(), 'Sandbox threads survived cleanup'

# Child RLM histories are lists too; origin must not be inferred from prompt shape.
from rlm.clients.openai import OpenAIClient
from rlm.core.types import ModelUsageSummary
answers = iter(['```repl\nplain = llm_query("Analyze a direct text subtask")\n'
                'child = rlm_query("Analyze a small text subtask")\n'
                'answer["content"] = {"solver.py":"x=2\\n"}; answer["ready"] = True\n```',
                'direct analysis',
                '```repl\nanswer["content"] = "child analysis"; answer["ready"] = True\n```'])
with mock_patch.object(OpenAIClient, 'completion', side_effect=lambda *args, **kwargs: next(answers)), \
     mock_patch.object(OpenAIClient, 'get_last_usage', return_value=ModelUsageSummary(1,1,1)):
    result = agent.solve({'id':'mock','repository':'mock','split':'train', 'prompt':'Repair solver',
                          'files':{'solver.py':'x=1\n'}, 'editable':['solver.py']},
                         'http://127.0.0.1:1', depth=2, calls=4, seconds=10)
assert result.get('patch') == {'solver.py':'x=2\n'}, result.get('error')
assert [call['kind'] for call in result['trace']] == ['root','subcall','subcall'], result['trace']
assert all(isinstance(call['messages'], list) for call in result['trace'])
assert result['calls'] == result['output_tokens'] == 3, 'Subcalls must share the root budget'
assert not any(t.name.startswith('local-rlm-') for t in threading.enumerate()), 'Sandbox thread leaked'
dev = next(task for task in json.loads(agent.TASKS.read_text()) if task['split'] == 'dev')
with tempfile.TemporaryDirectory() as directory:
    output = Path(directory) / 'report.json'
    row = {'id':dev['id'], 'repository':dev['repository'], 'split':'dev', 'passed':False,
           'patch':dev['reference_patch'], 'calls':1, 'elapsed_s':.1}
    with mock_patch.object(sys, 'argv', ['recursive_agent.py', '--task', dev['id'], '--output', str(output)]), \
         mock_patch.object(agent, 'solve', return_value=row), \
         mock_patch.object(agent, 'grade', side_effect=TimeoutError('mock grader timeout')):
        agent.main()
    saved = json.loads(output.read_text())['tasks'][0]
    assert not saved['passed'] and 'mock grader timeout' in saved['feedback']
print('PASS: patch paths, shared budgets, persistent isolated state, output and time bounds')
