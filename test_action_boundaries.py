"""Action context and executable/final response boundaries of the pinned runner."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import recursive_agent as agent
from repl_actions import Workspace


class ContextBindingsTests(unittest.TestCase):
    def test_text_subtask_cannot_invent_an_executable_turn_from_its_own_text(self):
        budget = agent.Budget(calls=12)
        budget.reserve(100)
        messages = budget.action_prompt('Turn 1/8: Return the requested analysis.', 'rlm', 'subcall')
        self.assertEqual(budget.phase, 'text')
        self.assertIn('No REPL is available', messages[-1]['content'])

    def test_parent_and_child_context_aliases_are_typed_and_cannot_change_canonical_task(self):
        task = dict(task='Repair the value.', files={'solver.py': 'x=1\n'}, editable=['solver.py'])
        with tempfile.TemporaryDirectory() as directory:
            for child, payload in ((False, task), (True, json.dumps(task))):
                workspace = Workspace(payload, {}, root=Path(directory)/str(child))
                bindings = workspace.bindings()
                self.assertIsInstance(bindings.get('context'), dict)
                self.assertIs(bindings['context'], bindings['task'])
                self.assertEqual(bindings['is_child'], child)
                bindings['context']['files']['solver.py'] = 'pretend repair'
                self.assertEqual(workspace.task, task)
                self.assertEqual(workspace._read('solver.py'), 'x=1\n')


@unittest.skipUnless(importlib.util.find_spec('rlm'), 'Pinned upstream required')
class PromptBoundaryTests(unittest.TestCase):
    def solve(self, replies, task=None, **kwargs):
        from rlm.clients.openai import OpenAIClient
        from rlm.core.types import ModelUsageSummary
        from test_agent_curriculum import LocalSandbox
        task = task or dict(id='boundary', repository='fixture', split='train', prompt='Repair the value.',
                            files={'solver.py': 'x=1\n'}, editable=['solver.py'])
        responses = iter(replies)
        with patch.object(agent, 'Sandbox', LocalSandbox), \
             patch.object(OpenAIClient, 'completion', side_effect=lambda *a, **k: next(responses)), \
             patch.object(OpenAIClient, 'get_last_usage', return_value=ModelUsageSummary(1, 1, 1)):
            return agent.solve(task, 'http://127.0.0.1:8090', calls=12, seconds=20,
                               action_api=True, **kwargs)

    def test_actual_parent_and_child_setup_show_task_and_true_role(self):
        row = self.solve([
            "```repl\nchild=rlm_query(__import__('json').dumps(context))\n"
            "answer['content']={'solver.py':'x=2\\n'};answer['ready']=True\n```",
            "```repl\nanswer['content']='child';answer['ready']=True\n```"], depth=2, instruction='KEEP_THIS_USER_RULE')
        self.assertEqual([c['kind'] for c in row['trace']], ['root', 'subcall'])
        for call, role in zip(row['trace'], ('Root REPL', 'Child REPL')):
            content = call['messages'][1]['content']
            self.assertIn(role, content)
            self.assertIn('Repair the value.', content)
            self.assertIn('x=1', content)
            self.assertNotIn('100k', str(call['messages']))
            self.assertNotIn('context is a str', str(call['messages']))
            self.assertNotIn('You have not interacted', str(call['messages']))
            self.assertIn('KEEP_THIS_USER_RULE', call['messages'][0]['content'])
            if role == 'Child REPL':
                self.assertNotIn('proposal = delegate', call['messages'][0]['content'])
                self.assertIn('You are the child REPL', call['messages'][0]['content'])
            else:
                self.assertIn('proposal = delegate', call['messages'][0]['content'])

    def test_last_turn_and_final_response_have_different_execution_contracts(self):
        row = self.solve(['```repl\npass\n```']*8 + ['{"solver.py":"x=2\\n"}'],
                         mode='baseline')
        self.assertEqual(len(row['trace']), 9)
        self.assertIn('last executable turn', row['trace'][7]['messages'][-1]['content'])
        final = row['trace'][-1]
        self.assertIn('No code will execute', final['messages'][0]['content'])
        self.assertIn('REPL turns remaining: 0.', final['messages'][-1]['content'])
        self.assertNotIn('When using the REPL', final['messages'][-1]['content'])
        self.assertEqual(final['phase'], 'final')
        self.assertEqual(len(row['executions']), 8)
        self.assertEqual(row['patch'], {'solver.py': 'x=2\n'})
        self.assertFalse(agent.tested_repl(dict(row, passed=True)))

    def test_nested_child_final_phase_does_not_leak_into_parent_turn(self):
        row = self.solve(["```repl\nchild=rlm_query(__import__('json').dumps(context))\n```"] +
                         ['```repl\npass\n```']*8 + ['{"solver.py":"x=2\\n"}',
                         "```repl\nanswer['content']={'solver.py':'x=2\\n'};answer['ready']=True\n```"], depth=2)
        self.assertEqual(len(row['trace']), 11)
        self.assertEqual(row['trace'][9]['phase'], 'final')
        last = row['trace'][-1]
        self.assertEqual((last['kind'], last['phase']), ('root', 'repl'))
        self.assertIn('including this response): 7 of 8.', last['messages'][-1]['content'])
        self.assertFalse(agent.FINAL_RESPONSE.get())

    def test_final_phase_resets_after_api_error(self):
        def responses():
            yield from ['```repl\npass\n```']*8
            raise RuntimeError('Injected final API failure')
        with self.assertRaisesRegex(RuntimeError, 'Infrastructure failure'):
            self.solve(responses(), mode='baseline')
        self.assertFalse(agent.FINAL_RESPONSE.get())

    def test_requirements_are_complete_and_source_preview_is_explicitly_bounded(self):
        requirement = 'Repair.'*500+'REQUIREMENTS_END'
        source = 'x=1\n'+'#'*1700+'SOURCE_END'
        task = dict(id='preview', repository='fixture', split='train', prompt=requirement,
                    files={'solver.py': source}, editable=['solver.py'], checks='PRIVATE_CHECK_SENTINEL',
                    reference_patch={'solver.py': 'PRIVATE_REFERENCE_SENTINEL'})
        row = self.solve(["```repl\nanswer['content']={'solver.py':'x=2\\n'};answer['ready']=True\n```"], task=task)
        content = row['trace'][0]['messages'][1]['content']
        self.assertIn(requirement, content)
        self.assertIn('Preview truncated', content)
        self.assertNotIn('SOURCE_END', content)
        self.assertNotIn('PRIVATE_CHECK_SENTINEL', content)
        self.assertNotIn('PRIVATE_REFERENCE_SENTINEL', content)


if __name__ == '__main__':
    unittest.main()
