"""Executed authored lessons; fake only the Docker boundary in portable checks."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import traceback
import unittest
import warnings
from unittest.mock import patch

import agent_curriculum as curriculum

REVISION = '4e0c13aab45cf2e0371208416e9b5e175f3c4361'


class CurriculumTests(unittest.TestCase):
    def test_token_preflight_rejects_missing_steps_and_oversized_masked_mistakes(self):
        from test_qwen35_training import Tokenizer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'traces').mkdir()
            row = dict(task_id='lesson:direct-source:action-0', language='python', split='train',
                example_kind='executed_repl_repair', origin='root',
                messages=[dict(role='user',content='Inspect'),dict(role='assistant',content='```repl\nprint(1)\n```')],
                provenance=dict(dataset='local/authored-agent-repairs',revision=REVISION,source_row_sha256='a'*64),
                verification=dict(kind='local_execution',passed=True))
            curriculum.write_dataset(root,[row],dict(sources={}))
            call=dict(messages=row['messages'][:-1],response=row['messages'][-1]['content'],kind='root')
            record=dict(task_id='lesson',variant='direct-source',rows=[row],result=dict(trace=[call],calls=1))
            path=root/'traces/lesson.json'
            path.write_text(json.dumps(record))
            self.assertTrue(curriculum.verify_tokens(root,REVISION,Tokenizer())['passed'])
            for damaged in (dict(record,rows=[]),dict(record,result=dict(trace=[],calls=1)),
                    dict(record,result=dict(trace=[dict(call,response='changed')],calls=1)),
                    dict(record,result=dict(trace=[dict(call,kind='subcall')],calls=1)),
                    dict(record,result=dict(trace=[call,dict(call,response='x'*1025)],calls=2))):
                path.write_text(json.dumps(damaged))
                with self.assertRaisesRegex(ValueError,'Complete trajectories'):
                    curriculum.verify_tokens(root,REVISION,Tokenizer())
                self.assertFalse(json.loads((root/'feasibility.json').read_text())['passed'])

    def test_only_twenty_train_families_and_private_checks_stay_private(self):
        tasks = curriculum.training_tasks()
        self.assertEqual(len(tasks), 20)
        self.assertEqual({task['split'] for task in tasks}, {'train'})
        for original in tasks:
            for visible in (True, False):
                task = curriculum.task_variant(original, visible)
                self.assertEqual('test_visible.py' in task['files'], visible)
                self.assertNotIn(original['checks'], '\n'.join(task['files'].values()))
                self.assertNotIn('reference_patch', task)
                if not visible:
                    self.assertNotIn('Run test_visible.py', task['prompt'])
        with self.assertRaises(ValueError):
            curriculum.task_variant(dict(tasks[0], split='dev'), True)

    def test_script_fits_call_budget_and_parent_uses_child_response(self):
        original = curriculum.training_tasks()[0]
        for recursive in (False, True):
            for visible in (False, True):
                task = curriculum.task_variant(original, visible)
                actions = curriculum.actions(task, original['reference_patch'], recursive,
                                              recovery=not recursive and visible)
                self.assertLessEqual(len(actions), 8)
                valid = [action for action in actions if not action.get('invalid')]
                self.assertEqual(sum(a['kind'] == 'subcall' for a in valid), 2 if recursive else 0)
                self.assertEqual(sum('subprocess.run' in a['response'] for a in valid), 2 if visible else 0)
                if recursive:
                    root = '\n'.join(a['response'] for a in valid if a['kind'] == 'root')
                    self.assertIn('json.loads(child_reply)', root)
                    self.assertNotIn(repr(original['reference_patch']), root)
                for action in valid:
                    compile(action['response'][8:-4], '<lesson>', 'exec')

    def test_manifest_and_anchors_preserve_execution_provenance(self):
        from qwen35_train import load_verified
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(task_id=f'anchor-{i}', language='go', split='train',
                messages=[dict(role='user', content='Do task'), dict(role='assistant', content='done')],
                provenance=dict(dataset='verified/source', revision=REVISION, source_row_sha256='a'*64),
                verification=dict(kind='local_execution', passed=True)) for i in range(35)]
            curriculum.write_dataset(root, rows, {'construction': 'test'})
            selected = curriculum.anchors(root/'train.jsonl')
            self.assertEqual(len(selected), 30)
            self.assertEqual(selected, rows[:30])
            loaded, _ = load_verified(root/'train.jsonl', root/'train.jsonl.manifest.json')
            self.assertEqual(loaded, rows)
            manifest = root/'train.jsonl.manifest.json'
            value = json.loads(manifest.read_text()); value['verification'] = 'publisher_execution'
            manifest.write_text(json.dumps(value))
            with self.assertRaises(ValueError): curriculum.anchors(root/'train.jsonl')


class LocalSandbox:
    """Execute trusted fixture cells locally; the real upstream RLM still routes children."""
    def __init__(self, context_payload, subcall_fn=None, depth=0, budget=None, **kwargs):
        self.temporary = tempfile.TemporaryDirectory()
        self.root, self.depth = Path(self.temporary.name), depth
        self.budget, self.workspace_hashes = budget, {}
        def query(prompt):
            first = len(budget.executions)
            response = subcall_fn(prompt).response
            budget.queries.append(dict(recursive=True, response=response,
                execution_start=first, execution_end=len(budget.executions)))
            return response
        self.state = {'context': context_payload, 'answer': {'content': '', 'ready': False},
                      'rlm_query': query}
        if isinstance(context_payload, dict):
            for name, content in context_payload['files'].items():
                path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)

    def execute_code(self, code):
        from rlm.core.types import REPLResult
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code.replace('/workspace', str(self.root)), self.state, self.state)
        except BaseException:
            traceback.print_exc(file=stderr)
        answer = self.state['answer']
        final = answer['content'] if answer['ready'] else None
        if isinstance(final, dict):
            self.workspace_hashes = {name: hashlib.sha256((self.root/name).read_bytes()).hexdigest()
                                     for name in final if (self.root/name).exists()}
        if self.budget:
            self.budget.executions.append(dict(depth=self.depth, code=code, stderr=stderr.getvalue(),
                final_answer=json.dumps(final) if isinstance(final, dict) else final))
        return REPLResult(stdout=stdout.getvalue(), stderr=stderr.getvalue(), locals={},
                          final_answer=json.dumps(final) if isinstance(final, dict) else final)

    def cleanup(self): self.temporary.cleanup()


@unittest.skipUnless(importlib.util.find_spec('rlm'), 'Pinned upstream environment required')
class ExecutedTests(unittest.TestCase):
    def test_four_variants_keep_recovery_context_and_mask_invalid_target(self):
        import recursive_agent as agent
        from test_rlm_tasks import run
        original = curriculum.training_tasks()[0]
        def grade(task, changes):
            return dict(passed=run({**task['files'], **changes}, task['checks']).returncode == 0)
        with warnings.catch_warnings(), patch.object(agent, 'Sandbox', LocalSandbox), patch.object(agent, 'grade', grade):
            warnings.simplefilter('ignore', ResourceWarning)
            records = [curriculum.trajectory(original, recursive, visible, REVISION)
                       for recursive in (False, True) for visible in (True, False)]
        rows = [row for record in records for row in record['rows']]
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({row['task_id'] for row in rows}), 20)
        self.assertEqual(sum(row['origin'] == 'subcall' for row in rows), 4)
        self.assertTrue(any('No REPL command was executed' in str(row['messages']) for row in rows))
        self.assertTrue(all(row['messages'][-1]['content'].startswith('```repl\n') for row in rows))
        self.assertTrue(all(original['checks'] not in str(row['messages']) for row in rows))
        self.assertTrue(all(row['verification']['passed'] for row in rows))

    def test_failed_reference_does_not_export_rows(self):
        import recursive_agent as agent
        original = copy.deepcopy(curriculum.training_tasks()[0])
        original['reference_patch']['solver.py'] = 'def repair(a,b): return []\n'
        with warnings.catch_warnings(), patch.object(agent, 'Sandbox', LocalSandbox), patch.object(agent, 'grade', return_value={'passed': False}):
            warnings.simplefilter('ignore', ResourceWarning)
            with self.assertRaises(ValueError): curriculum.trajectory(original, False, True, REVISION)

    def test_infrastructure_error_blocks_even_a_passing_patch(self):
        import recursive_agent as agent
        from test_rlm_tasks import run
        original = curriculum.training_tasks()[0]
        solve = agent.solve
        def interrupted(*args, **kwargs):
            return dict(solve(*args, **kwargs), infrastructure_errors=['Injected child API failure'])
        def grade(task, changes):
            return dict(passed=run({**task['files'], **changes}, task['checks']).returncode == 0)
        with warnings.catch_warnings(), patch.object(agent, 'Sandbox', LocalSandbox), \
             patch.object(agent, 'grade', grade), patch.object(agent, 'solve', side_effect=interrupted):
            warnings.simplefilter('ignore', ResourceWarning)
            with self.assertRaises(ValueError): curriculum.trajectory(original, False, False, REVISION)


@unittest.skipUnless(os.environ.get('LCA_AGENT_NATIVE_TESTS') == '1', 'Explicit native Docker check')
class NativeTests(unittest.TestCase):
    def test_real_sandbox_four_variants(self):
        original = curriculum.training_tasks()[0]
        records = [curriculum.trajectory(original, recursive, visible, REVISION)
                   for recursive in (False, True) for visible in (True, False)]
        self.assertEqual(sum(len(record['rows']) for record in records), 20)


if __name__ == '__main__': unittest.main()
