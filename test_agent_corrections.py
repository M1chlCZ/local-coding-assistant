"""Recorded conditioning history must not become a new positive target."""
import copy
import importlib.util
import json
import unittest
from unittest.mock import patch

import agent_corrections as corrections
import recursive_agent as agent


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        return {'input_ids': self.encode(json.dumps(messages))}


@unittest.skipUnless(importlib.util.find_spec('rlm'), 'Pinned upstream environment required')
class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.task = dict(id='fixture', repository='fixture', split='train', prompt='Original wording.',
            files={'solver.py': 'x=0\n'}, editable=['solver.py'], checks='assert private_check',
            reference_patch={'solver.py': 'x=2\n'})
        self.messages = [dict(role='user', content='The exact recorded prompt')]
        self.prefix = dict(kind='root', messages=self.messages, response='```repl\nshow()\n```',
            input_tokens=len(json.dumps(self.messages)), output_tokens=20)
        self.receipt = corrections.prepare_context(Tokenizer(),
            dict(messages=self.messages, max_completion_tokens=1024), 8192, 1024)
        self.source = dict(trace=[self.prefix], context_windows=[dict(self.receipt,
            requested_messages=self.messages, kind='root')])
        self.plan = [dict(kind='root', response='```repl\nfinish()\n```')]

    def run_replay(self, changed=False):
        from rlm.clients.openai import OpenAIClient
        from types import SimpleNamespace
        def solve(task, base, **kwargs):
            self.assertEqual(task['prompt'], self.task['prompt'])
            self.assertNotIn('reference_patch', task)
            self.assertNotIn('diagnostic', task)
            self.assertTrue(kwargs['context_window'])
            budget = agent.Budget(**{k: v for k,v in kwargs.items() if k in ('calls','output_tokens','seconds')})
            client = SimpleNamespace(sampling_args={'max_completion_tokens':1024})
            trace = []
            for i in range(2):
                prompt = copy.deepcopy(self.messages)
                if changed and not i:
                    prompt[0]['content'] += ' silently changed'
                maximum = budget.reserve(1024)
                effective = budget.prepare_prompt(prompt, base, maximum)
                reply = OpenAIClient.completion(client, effective)
                used = OpenAIClient.get_last_usage(client)
                trace.append(dict(messages=effective,response=reply,kind='root',
                    input_tokens=used.total_input_tokens,output_tokens=used.total_output_tokens))
            return dict(trace=trace, calls=2, context_windows=budget.context_windows,
                patch=self.task['reference_patch'], workspace_hashes={}, executions=[])
        with patch.object(agent, 'solve', solve), patch.object(agent, 'grade', return_value={'passed': True}), \
             patch.object(corrections, 'verify_result'), \
             patch.object(corrections, 'response_tokens', return_value=20):
            return corrections.replay(self.task, False, self.source, 1, self.plan, 'a'*40, Tokenizer())

    def test_exact_prompt_and_real_accounting_mask_the_recorded_prefix(self):
        record = self.run_replay()
        self.assertEqual(record['target_indices'], [1])
        self.assertEqual(len(record['rows']), 1)
        self.assertEqual(record['rows'][0]['messages'][-1]['content'], self.plan[0]['response'])
        self.assertEqual(record['result']['trace'][0]['output_tokens'], 20)
        self.assertEqual(record['result']['trace'][1]['input_tokens'], self.prefix['input_tokens'])
        self.assertTrue(record['prefix_reused'])

    def test_changed_conditioning_prompt_is_rejected_before_export(self):
        with self.assertRaisesRegex(ValueError, 'Recorded requested prefix'):
            self.run_replay(changed=True)

    def test_training_length_cannot_replace_the_recorded_runtime_context(self):
        self.source['context_windows'][0]['context_tokens'] = 4096
        with self.assertRaisesRegex(ValueError, 'Recorded effective context'):
            self.run_replay()

    def test_only_registered_training_tasks_and_nonempty_bounded_prefixes(self):
        for task, length in [(dict(self.task,split='dev'),1), (self.task,0), (self.task,2)]:
            with self.assertRaises(ValueError):
                corrections.replay(task,False,self.source,length,self.plan,'a'*40,Tokenizer())


if __name__ == '__main__':
    unittest.main()
