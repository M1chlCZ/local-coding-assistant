"""Corrective imitation must execute and mask actual student mistakes."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import agent_recovery as recovery
import recursive_agent as agent
from agent_curriculum import training_tasks
from test_agent_curriculum import LocalSandbox, REVISION
from test_rlm_tasks import run


@unittest.skipUnless(importlib.util.find_spec('rlm'), 'Pinned upstream environment required')
class RecoveryTests(unittest.TestCase):
    def test_real_error_context_with_only_verified_teacher_targets(self):
        from rlm.clients.openai import OpenAIClient
        from rlm.core.types import ModelUsageSummary
        task = training_tasks()[0]
        for recursive in (False, True):
            student = iter(["```repl\nprint(context['task'])\n```",
                            "```repl\nraise ValueError('student test failure')\n```"])
            def grade(t, changes):
                return dict(passed=run({**t['files'], **changes}, t['checks']).returncode == 0)
            with patch.object(agent, 'Sandbox', LocalSandbox), patch.object(agent, 'grade', grade), \
                 patch.object(OpenAIClient, 'completion', side_effect=lambda *a, **k: next(student)), \
                 patch.object(OpenAIClient, 'get_last_usage', return_value=ModelUsageSummary(1,1,1)):
                record = recovery.trajectory(task, recursive, 2, REVISION)
            self.assertTrue(record['accepted'], record.get('reason'))
            self.assertEqual(record['teacher_calls'], list(range(2, record['result']['calls'])))
            self.assertLessEqual(record['result']['calls'], 8)
            self.assertTrue(any('student test failure' in str(r['messages'][:-1]) for r in record['rows']))
            self.assertFalse(any('student test failure' in r['messages'][-1]['content'] for r in record['rows']))
            self.assertTrue(all(task['checks'] not in str(r['messages']) for r in record['rows']))
            if recursive:
                self.assertTrue(recovery.recursive_success(record['result'], task))
                self.assertTrue(all(recovery.RECURSIVE in r['messages'][0]['content'] for r in record['rows']))

    def test_nontraining_or_failed_corrections_cannot_be_lessons(self):
        task = training_tasks()[0]
        with self.assertRaises(ValueError):
            recovery.trajectory(dict(task, split='dev'), False, 0, REVISION)
        bad = copy.deepcopy(task)
        bad['reference_patch'] = {'solver.py': 'def repair(a,b): return []\n'}
        with patch.object(agent, 'grade', return_value={'passed': False}), \
             patch.object(agent, 'Sandbox', LocalSandbox):
            record = recovery.trajectory(bad, False, 0, REVISION)
        self.assertFalse(record['accepted'])
        self.assertEqual(record['rows'], [])


class RecoveryPreflightTests(unittest.TestCase):
    def test_partial_collection_and_changed_targets_are_rejected(self):
        from test_qwen35_training import Tokenizer
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'traces').mkdir()
            with self.assertRaisesRegex(ValueError, 'finish every declared'):
                recovery.finalize(folder, REVISION, folder/'unused', Tokenizer())
            call = dict(messages=[dict(role='user',content='task')], response='```repl\npass\n```')
            record = dict(accepted=True, teacher_calls=[0], result=dict(trace=[call],calls=1),
                          rows=[dict(messages=[dict(role='assistant',content='unexecuted target')])])
            for i in range(4):
                (folder/f'traces/{i}.json').write_text(json.dumps(record))
            with patch.object(recovery, 'training_tasks', return_value=[{}]), \
                 self.assertRaisesRegex(ValueError, 'no longer match the executed trace'):
                recovery.finalize(folder, REVISION, folder/'unused', Tokenizer())


if __name__ == '__main__': unittest.main()
