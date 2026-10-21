"""Protocol checks use scripted actions, never claim learned model ability."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from repl_actions import Workspace
from research import tool_protocol as protocol


def response(tool, path='', text='', finish='stop'):
    return dict(choices=[dict(finish_reason=finish, message=dict(
        content=json.dumps(dict(tool=tool, path=path, text=text))))],
        usage=dict(prompt_tokens=20, completion_tokens=10))


class ProtocolTests(unittest.TestCase):
    def test_read_action_displays_file_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            source = 'value = "quoted\\\\path"\n'
            task = dict(task='Inspect.', editable=['solver.py'], files={'solver.py': source})
            namespace = Workspace(task, {}, root=directory).bindings()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(protocol.action_code(response('read', 'solver.py'), False), namespace)
            self.assertEqual(output.getvalue(), source)

    def test_summary_requires_all_matched_tasks_and_never_authorizes_learning(self):
        from research.tool_protocol_compare import summarize
        entries = [dict(condition='ordinary', task=dict(id='one'))]
        rows = [dict(id='one', interface=mode, passed=mode != 'typed',
                     tested_repl=mode != 'typed', recursive_passed=False, calls=2,
                     output_tokens=20, elapsed_s=1, protocol_errors=[])
                for mode in ('fenced', 'typed', 'constrained')]
        with self.assertRaises(ValueError):
            summarize(rows[:-1], entries)
        with self.assertRaises(ValueError):
            summarize(rows+[rows[0]], entries)
        summary = summarize(rows, entries)
        self.assertEqual(summary['versus_fenced']['typed']['functional_losses'], ['one'])
        self.assertFalse(summary['learning_gain_established'])
        self.assertFalse(summary['full_loop_authorized'])

    def test_literal_source_roundtrip_without_executing_it_in_dispatch(self):
        source = "def value():\n    return " + repr('a"\\b\n</tool_call>') + '\n'
        code = protocol.action_code(response('write', 'solver.py', source), False)
        received = []
        exec(code, {'write': lambda *args: received.append(args)})
        self.assertEqual(received, [('solver.py', source)])

    def test_rejects_truncation_duplicate_fields_unknown_actions_and_child_delegation(self):
        invalid = [response('finish', finish='length'), response('unknown'),
                   response('finish', text='pretend success'), response('read', '../escape.py')]
        duplicate = response('finish')
        duplicate['choices'][0]['message']['content'] = (
            '{"tool":"write","tool":"finish","path":"","text":""}')
        invalid.append(duplicate)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.action_code(value, False)
        with self.assertRaises(ValueError):
            protocol.action_code(response('delegate', text='repair'), True)

    def test_failed_check_or_later_edit_cannot_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            answer = {}
            task = dict(task='Make value return 2', editable=['solver.py'],
                        files={'solver.py': 'def value():\n    return 1\n'})
            workspace = Workspace(task, answer, root=directory)
            namespace = workspace.bindings()
            with contextlib.redirect_stdout(io.StringIO()):
                exec(protocol.action_code(response('check', text='from solver import value\nassert value()==2'), False), namespace)
                exec(protocol.action_code(response('write', 'solver.py', 'def value():\n    return 2\n'), False), namespace)
                with self.assertRaises(ValueError):
                    exec(protocol.action_code(response('finish'), False), namespace)
                exec(protocol.action_code(response('check', text='from solver import value\nassert value()==2'), False), namespace)
                exec(protocol.action_code(response('finish'), False), namespace)
            self.assertTrue(answer['ready'])
            self.assertEqual(workspace.checks[-1]['assertions'], 1)

    @unittest.skipUnless(importlib.util.find_spec('rlm'), 'Pinned upstream and Docker required')
    def test_real_child_returns_only_after_checks_and_parent_rechecks(self):
        from agent_pilot import recursive_success, tested_repl
        from recursive_agent import grade
        task = dict(id='scripted-protocol', repository='scripted', split='dev',
                    prompt='Make value() return 2.', files={'solver.py':'def value():\n    return 1\n'},
                    editable=['solver.py'], checks='from solver import value\nassert value()==2')
        checks = 'from solver import value\nassert value()==2'
        actions = iter([response('show'), response('delegate', text=task['prompt']),
                        response('check', text=checks),
                        response('write', 'solver.py', 'def value():\n    return 2\n'),
                        response('check', text=checks), response('finish'),
                        response('apply'), response('check', text=checks), response('finish')])
        def prepare(budget, messages, base, maximum):
            return messages
        with patch.object(protocol, 'request', side_effect=lambda *a, **k: next(actions)), \
             patch.object(protocol.Budget, 'prepare_prompt', prepare):
            result = protocol.solve(task, 'http://127.0.0.1:8090', recursive=True, seconds=60)
        self.assertFalse(result['infrastructure_errors'])
        result['passed'] = grade(task, result['patch'])['passed']
        self.assertTrue(tested_repl(result))
        self.assertTrue(recursive_success(result, task))
        self.assertEqual(result['calls'], 9)
        self.assertEqual(len(result['queries']), 1)

    @unittest.skipUnless(importlib.util.find_spec('rlm'), 'Pinned upstream and Docker required')
    def test_exhausted_child_never_produces_a_fallback_patch_and_outage_propagates(self):
        task = dict(id='failure', repository='scripted', split='dev', prompt='Repair',
                    files={'solver.py':'x=1'}, editable=['solver.py'])
        def prepare(budget, messages, base, maximum):
            return messages
        actions = iter([response('delegate', text='repair'), response('finish')])
        with patch.object(protocol, 'request', side_effect=lambda *a, **k: next(actions)), \
             patch.object(protocol.Budget, 'prepare_prompt', prepare):
            result = protocol.solve(task, 'http://127.0.0.1:8090', recursive=True, calls=2, seconds=30)
        self.assertNotIn('patch', result)
        self.assertFalse(result['queries'])
        self.assertEqual(result['calls'], 2)
        with patch.object(protocol, 'request', side_effect=[
                response('delegate', text='repair'), ConnectionError('child server unavailable')]), \
             patch.object(protocol.Budget, 'prepare_prompt', prepare):
            with self.assertRaisesRegex(RuntimeError, 'Infrastructure'):
                protocol.solve(task, 'http://127.0.0.1:8090', recursive=True, calls=2, seconds=30)


if __name__ == '__main__':
    unittest.main()
