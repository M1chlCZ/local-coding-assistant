"""Exact prompt budgeting without cutting a task, an action, or its feedback."""
import copy
import importlib.util
import json
import threading
import unittest
from http.server import HTTPServer
from urllib.request import Request, urlopen
from unittest.mock import patch

import qwen35_server as server
import recursive_agent as agent


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False,
                              return_dict=True)
        # Include role/template overhead and make characters unequal to tokens.
        return {'input_ids': [0] * (5 + sum(4 + len(m['content'].encode()) for m in messages))}


def messages():
    return [
        dict(role='system', content='Keep the repository task and sandbox rules.'),
        dict(role='user', content='Repair both files; context contains the sources.'),
        dict(role='user', content='Inspect context first.'),
        dict(role='assistant', content='old code ' * 60),
        dict(role='user', content='REPL output:\n' + 'old traceback ' * 100),
        dict(role='user', content='Turn 2/8:'),
        dict(role='assistant', content='write("a.py", "answer = 42")'),
        dict(role='user', content='REPL output:\nCheck exit: 0\n✓'),
        dict(role='user', content='Turn 3/8:'),
    ]


class ContextTests(unittest.TestCase):
    def fit(self, prompt, context=600, maximum=100):
        return server.prepare_context(Tokenizer(), dict(messages=prompt,
            max_completion_tokens=maximum), context, 1024)

    def test_complete_old_turn_removed_and_newest_feedback_preserved(self):
        original = messages()
        snapshot = copy.deepcopy(original)
        result = self.fit(original)
        self.assertTrue(result['fits'])
        self.assertEqual(result['dropped_turns'], 1)
        self.assertEqual(result['messages'][:3], original[:3])
        self.assertEqual(result['messages'][-3:], original[-3:])
        self.assertLessEqual(result['prompt_tokens'] + 100, 600)
        self.assertGreater(result['original_prompt_tokens'] + 100, 600)
        self.assertEqual(original, snapshot)
        self.assertEqual(self.fit(original), result)

    def test_under_limit_unchanged_including_exact_boundary(self):
        original = messages()[:3]
        count = len(Tokenizer().apply_chat_template(original, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=True)['input_ids'])
        result = self.fit(original, context=count + 100)
        self.assertTrue(result['fits'])
        self.assertEqual(result['messages'], original)
        self.assertEqual(result['dropped_turns'], 0)

    def test_oversized_initial_or_latest_turn_cannot_be_cut(self):
        for prompt in [messages()[:3], messages()[-3:], messages()]:
            result = self.fit(prompt, context=110)
            self.assertFalse(result['fits'])
            self.assertEqual(result['messages'][-1], prompt[-1])
            self.assertGreater(result['prompt_tokens'] + 100, 110)

    def test_no_mid_history_system_instruction_is_discarded(self):
        original = messages()
        original.insert(4, dict(role='system', content='Additional invariant'))
        with self.assertRaises(ValueError):
            self.fit(original)

    def test_context_endpoint_does_not_generate_and_validates_input(self):
        calls = []
        http = HTTPServer(('127.0.0.1', 0), server.make_handler(
            lambda *a: calls.append(a), 'model', 'revision', False,
            prepare_context_fn=lambda data: server.prepare_context(Tokenizer(), data, 600, 1024)))
        thread = threading.Thread(target=http.serve_forever)
        thread.start()
        try:
            request = Request(f'http://127.0.0.1:{http.server_port}/context',
                json.dumps(dict(messages=messages(), max_completion_tokens=100)).encode(),
                {'Content-Type': 'application/json'})
            with urlopen(request) as response:
                result = json.load(response)
            self.assertTrue(result['fits'])
            self.assertEqual(calls, [])
        finally:
            http.shutdown()
            thread.join()
            http.server_close()

    def test_agent_preserves_full_and_effective_prompts_and_never_retries_exhaustion(self):
        http = HTTPServer(('127.0.0.1', 0), server.make_handler(
            lambda *a: self.fail('Preflight must not generate'), 'model', 'revision', False,
            prepare_context_fn=lambda data: server.prepare_context(Tokenizer(), data, 600, 1024)))
        thread = threading.Thread(target=http.serve_forever)
        thread.start()
        try:
            base = f'http://127.0.0.1:{http.server_port}'
            budget = agent.Budget(calls=12)
            maximum = budget.reserve(100)
            effective = budget.prepare_prompt(messages(), base, maximum)
            self.assertEqual(budget.context_windows[0]['requested_messages'], messages())
            self.assertEqual(budget.context_windows[0]['messages'], effective)
            self.assertLess(len(effective), len(messages()))
            recorded = copy.deepcopy(budget.context_windows[0])
            effective.append(dict(role='assistant', content='next action'))
            effective[0]['content'] = 'changed by caller'
            self.assertEqual(budget.context_windows[0], recorded)
            with self.assertRaises(agent.ContextWindowExceeded):
                budget.prepare_prompt([dict(role='user', content='large task' * 100)], base, 100)
            self.assertEqual(len(budget.context_windows), 2)
            with self.assertRaises(agent.ContextWindowExceeded):
                budget.reserve(100)
            self.assertEqual(budget.calls, 1)
        finally:
            http.shutdown()
            thread.join()
            http.server_close()

    def test_frozen_policy_reaches_both_modes_and_exhaustion_is_not_outage(self):
        from agent_pilot import evaluate_one
        row = dict(id='task', passed=False, elapsed_s=1, infrastructure_errors=[],
                   error='ContextWindowExceeded: Task and latest turn exceed context')
        for condition in ('ordinary', 'recursive'):
            with patch('recursive_agent.solve', return_value=row) as solve, \
                 patch('polyglot_runtime.check_program', return_value={'passed': True}):
                result = evaluate_one(dict(condition=condition, context_window=True,
                    task=dict(id='task', repository='task', split='dev')))
            self.assertTrue(solve.call_args.kwargs['context_window'])
            self.assertFalse(result['passed'])

    @unittest.skipUnless(importlib.util.find_spec('rlm'), 'Pinned upstream environment required')
    def test_real_upstream_preflights_parent_child_and_stops_oversized_task(self):
        from rlm.clients.openai import OpenAIClient
        from rlm.core.types import ModelUsageSummary
        from test_agent_curriculum import LocalSandbox
        limit = [20000]
        http = HTTPServer(('127.0.0.1', 0), server.make_handler(
            lambda *a: self.fail('Preflight must not generate'), 'model', 'revision', False,
            prepare_context_fn=lambda data: server.prepare_context(Tokenizer(), data, limit[0], 1024)))
        thread = threading.Thread(target=http.serve_forever)
        thread.start()
        task = dict(id='context', repository='fixture', split='dev', prompt='Repair solver',
                    files={'solver.py': 'x=1'}, editable=['solver.py'])
        try:
            base = f'http://127.0.0.1:{http.server_port}'
            replies = iter(["```repl\nchild=rlm_query('inspect a small subtask')\n"
                            "answer['content']={'solver.py':'x=2'};answer['ready']=True\n```",
                            "```repl\nanswer['content']='child response';answer['ready']=True\n```"])
            with patch.object(agent, 'Sandbox', LocalSandbox), \
                 patch.object(OpenAIClient, 'completion', side_effect=lambda *a, **k: next(replies)), \
                 patch.object(OpenAIClient, 'get_last_usage', return_value=ModelUsageSummary(1, 1, 1)):
                row = agent.solve(task, base, depth=2, calls=4, seconds=10, context_window=True)
            self.assertEqual(row.get('patch'), {'solver.py': 'x=2'})
            self.assertEqual([c['kind'] for c in row['context_windows']], ['root', 'subcall'])
            self.assertEqual([c['messages'] for c in row['context_windows']],
                             [c['messages'] for c in row['trace']])
            for call in row['trace']:
                self.assertNotIn('As an RLM, you should act as an orchestrator',
                                 call['messages'][0]['content'])
                self.assertNotIn('Shared model calls remaining', str(call['messages']))
            limit[0] = 100
            with patch.object(agent, 'Sandbox', LocalSandbox), \
                 patch.object(OpenAIClient, 'completion') as completion:
                row = agent.solve(task, base, calls=4, seconds=10, context_window=True)
            completion.assert_not_called()
            self.assertTrue(row['context_exhausted'])
            self.assertEqual(len(row['context_windows']), 1)
            self.assertFalse(row['infrastructure_errors'])
            self.assertFalse(row.get('patch'))
        finally:
            http.shutdown()
            thread.join()
            http.server_close()


    @unittest.skipUnless(importlib.util.find_spec('rlm'), 'Pinned upstream environment required')
    def test_action_requests_show_live_shared_budget_before_context_preparation(self):
        from rlm.clients.openai import OpenAIClient
        from rlm.core.types import ModelUsageSummary
        from test_agent_curriculum import LocalSandbox
        http = HTTPServer(('127.0.0.1', 0), server.make_handler(
            lambda *a: self.fail('Preflight must not generate'), 'model', 'revision', False,
            prepare_context_fn=lambda data: server.prepare_context(Tokenizer(), data, 20000, 1024)))
        thread = threading.Thread(target=http.serve_forever); thread.start()
        task = dict(id='shared-budget', repository='fixture', split='train', prompt='Repair solver',
                    files={'solver.py': 'x=1'}, editable=['solver.py'])
        try:
            replies = iter(["```repl\nchild=rlm_query('inspect a small subtask')\n"
                            "answer['content']={'solver.py':'x=2'};answer['ready']=True\n```",
                            "```repl\nanswer['content']='child response';answer['ready']=True\n```"])
            with patch.object(agent, 'Sandbox', LocalSandbox), \
                 patch.object(OpenAIClient, 'completion', side_effect=lambda *a, **k: next(replies)), \
                 patch.object(OpenAIClient, 'get_last_usage', return_value=ModelUsageSummary(1, 1, 1)):
                row = agent.solve(task, f'http://127.0.0.1:{http.server_port}', depth=2, calls=4,
                                  seconds=10, action_api=True, context_window=True)
            self.assertEqual([c['kind'] for c in row['trace']], ['root', 'subcall'])
            for index, (call, window) in enumerate(zip(row['trace'], row['context_windows'])):
                self.assertIn(f'Shared model calls remaining (including this response): {4-index} of 4.',
                              call['messages'][-1]['content'])
                self.assertEqual(window['requested_messages'][-1], call['messages'][-1])
                self.assertEqual(window['messages'], call['messages'])
            self.assertIn('one later root call', row['trace'][1]['messages'][-1]['content'])
            self.assertEqual(row['calls'], 2)
        finally:
            http.shutdown(); thread.join(); http.server_close()

    def test_shared_budget_notice_is_transient_survives_eviction_and_keeps_hard_limit(self):
        original = messages()
        snapshot = copy.deepcopy(original)
        budget = agent.Budget(calls=2)
        budget.reserve(100)
        first = budget.action_prompt(original, 'baseline', 'root')
        self.assertNotIn('Delegate', first[-1]['content'])
        self.assertEqual(original, snapshot)
        budget.reserve(100)
        second = budget.action_prompt(original, 'rlm', 'subcall')
        self.assertEqual(len(second), len(original)+1)
        self.assertIn('including this response): 1 of 2.', second[-1]['content'])
        self.assertNotIn('including this response): 2 of 2.', second[-1]['content'])
        prepared = self.fit(second, context=1200)
        self.assertTrue(prepared['fits'])
        self.assertGreater(prepared['dropped_turns'], 0)
        self.assertEqual(prepared['messages'][-1], second[-1])
        with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
            budget.reserve(100)
        self.assertEqual(original, snapshot)


if __name__ == '__main__':
    unittest.main()
