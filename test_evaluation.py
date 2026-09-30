"""Check evaluation safeguards with: python test_evaluation.py."""
try:
    import evaluation
except ModuleNotFoundError:
    raise AssertionError("Evaluation safeguards have not been implemented yet") from None

import json
import ast
import subprocess
import sys
from unittest.mock import patch

for thinking in (False, True):
    with patch.object(evaluation, 'request', return_value={}) as api:
        evaluation.chat('unused', [], thinking=thinking)
        assert api.call_args.args[2]['chat_template_kwargs']['enable_thinking'] is thinking

command = evaluation.container_command("python:3.14-slim", interactive=True)
assert command[command.index("--network") + 1] == "none"
assert "--read-only" in command and "--cap-drop=ALL" in command
assert "--user=65534:65534" in command
assert not any(arg in command for arg in ["-v", "--volume", "--privileged"])
assert evaluation.extract_code("```python\ndef add(a,b): return a+b\n```\n") == "def add(a,b): return a+b"
message = {"tool_calls": [{"id": "call_1", "type": "function", "function": {
    "name": "write_file", "arguments": json.dumps({"path": "solver.py", "content": "def add(a,b): return a+b"})}}]}
assert evaluation.parse_tools(message)[0]["function"]["name"] == "write_file"
for arguments in [{"path": "../secret", "content": "x"}, {"path": "/etc/passwd", "content": "x"},
                  {"path": "solver.py"}, {"path": "solver.py", "content": 1}]:
    message["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments)
    try:
        evaluation.parse_tools(message)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Unsafe tool arguments accepted: {arguments}")
print("PASS: isolated execution and bounded tool access")

import benchmark
tasks = benchmark.load_tasks()
assert len(tasks) == 32 and len({t['task_id'] for t in tasks}) == 32
assert tasks == benchmark.load_tasks(), 'Benchmark subset must remain deterministic'
assert all(t['entry_point'] and t['prompt'] and t['test'] for t in tasks)
task = {'entry_point': 'solve', 'prompt': 'def helper(): return 3\n\ndef solve():\n    """Return three."""\n'}
for answer in ['```python\n    return helper()\n```', 'def solve(): return helper()']:
    assembled = benchmark.assemble_solution(task, answer)
    assert 'def helper()' in assembled
    ast.parse(assembled)
if '--execute' in sys.argv:
    for answer in ['```python\n    return helper()\n```', 'def solve(): return helper()']:
        assert evaluation.check_code(benchmark.assemble_solution(task, answer), 'assert solve() == 3')['passed']
    assert not evaluation.check_code("print('x' * 5000)", '')['passed'], 'Generated output must be capped'
    checked = evaluation.check_code('while True: pass', '', timeout=1)
    assert not checked['passed'] and 'timeout' in checked['feedback'].lower()
    assert not subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=local-coding-check-'], text=True).strip()

if '--execute' in sys.argv:
    code = "from pathlib import Path\nPath('/workspace/test_solver.py').write_text('pass\\n')\ndef unique_sorted(values): return sorted(set(values))\n"
    message = {'role': 'assistant', 'tool_calls': [{'id': 'tamper', 'type': 'function',
               'function': {'name': 'write_file', 'arguments': json.dumps({'path': 'solver.py', 'content': code})}}]}
    with patch.object(evaluation, 'chat', side_effect=[({'choices': [{'message': message}]}, 0),
                      ({'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]}, 0)]):
        result = evaluation.run_agent('unused')
        assert not result['passed'] and 'modified' in result['output']
    print('PASS: bounded output, timeout cleanup, and fixed-test integrity')
