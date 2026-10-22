"""No-test repository lessons must learn checks without inventing task context."""
import copy
import importlib.util
import unittest
import warnings
from unittest.mock import patch

import agent_curriculum as curriculum
import agent_multifile as multifile
from test_agent_tested import fixture
from test_rlm_tasks import run
from test_agent_curriculum import LocalSandbox, REVISION


class MultiFileTests(unittest.TestCase):
    def test_preparation_excludes_checks_that_accept_a_no_op(self):
        import recursive_agent as agent
        entry=fixture()
        entry['answer']=entry['answer'].replace('return x+1','return None')
        name=entry['family']
        entry['tests']='\n'.join(f'assert {name}({x}) is None' for x in range(7))
        value=multifile.task(entry,0)
        def grade(t,changes):return dict(passed=run({**t['files'],**changes},t['checks']).returncode==0)
        with patch.object(agent,'grade',grade):
            self.assertEqual(multifile.lesson_rejection(value),'private checks accept the broken caller')
            self.assertIsNone(multifile.lesson_rejection(multifile.task(fixture(),0)))
        with patch.object(agent,'grade',side_effect=RuntimeError('container unavailable')):
            with self.assertRaisesRegex(RuntimeError,'container unavailable'):
                multifile.lesson_rejection(value)

    @unittest.skipUnless(importlib.util.find_spec('rlm'),'Pinned upstream environment required')
    def test_real_actions_derive_checks_then_child_executes_both_file_repairs(self):
        import recursive_agent as agent
        task=multifile.task(fixture(),2)
        original=copy.deepcopy(task)
        def grade(t,changes):return dict(passed=run({**t['files'],**changes},t['checks']).returncode==0)
        for recursive in (False,True):
            with warnings.catch_warnings(),patch.object(agent,'Sandbox',LocalSandbox),patch.object(agent,'grade',grade):
                warnings.simplefilter('ignore',ResourceWarning)
                record=curriculum.trajectory(task,recursive,False,REVISION,plan=multifile.actions(task,recursive),
                    instruction=multifile.INSTRUCTION if recursive else multifile.ORDINARY,
                    mode='rlm' if recursive else 'baseline')
            self.assertEqual(task,original)
            self.assertTrue(record['result']['passed'])
            self.assertEqual(record['result']['patch'],task['reference_patch'])
            self.assertNotIn(task['checks'],str(record['rows']))
            if recursive:
                query=record['result']['queries'][0]
                child=record['executions'][query['execution_start']:query['execution_end']]
                self.assertIn('Check exit: 1',child[0]['stdout'])
                self.assertIn('Check exit: 0',child[1]['stdout'])
            else:self.assertFalse(record['result']['queries'])

    def test_ordinary_evaluation_disables_queries_and_uses_same_check_instructions(self):
        from agent_pilot import evaluate_one
        row=dict(id='x',passed=False,error='',infrastructure_errors=[])
        with patch('recursive_agent.solve',return_value=row) as solve, \
             patch('polyglot_runtime.check_program',return_value={'passed':True}):
            evaluate_one(dict(task={'id':'x'},condition='ordinary',strict_ordinary=True,
                              ordinary_instruction=multifile.ORDINARY))
        self.assertEqual(solve.call_args.kwargs['mode'],'baseline')
        self.assertEqual(solve.call_args.kwargs['instruction'],multifile.ORDINARY)

    def test_task_has_no_tests_and_each_edit_is_required(self):
        task = multifile.task(fixture(), 3)
        self.assertEqual(len(task['editable']), 2)
        self.assertGreater(len(task['files']), 2)
        self.assertFalse(any('test' in name for name in task['files']))
        self.assertNotIn('(77)', task['prompt'])
        self.assertNotIn('(77)', task['diagnostic'])
        self.assertIn('(77)', task['checks'])
        self.assertEqual(run({**task['files'], **task['reference_patch']}, task['checks']).returncode, 0)
        for omitted in task['editable']:
            partial = {p:s for p,s in task['reference_patch'].items() if p != omitted}
            self.assertNotEqual(run({**task['files'], **partial}, task['checks']).returncode, 0)
        self.assertNotEqual(run(task['files'], task['diagnostic']).returncode, 0)
        self.assertEqual(run({**task['files'], **task['reference_patch']}, task['diagnostic']).returncode, 0)
        self.assertNotIn('test_visible.py', str(curriculum.task_variant(task, False)))

    def test_child_gets_dependencies_without_private_checks_or_parent_gold(self):
        task = multifile.task(fixture(), 1)
        plan = multifile.actions(task, True)
        roots = '\n'.join(p['response'] for p in plan if p['kind'] == 'root')
        self.assertNotIn(repr(task['reference_patch']), roots)
        self.assertNotIn(task['checks'], str(plan))
        self.assertNotIn("context['files'][", roots)
        self.assertIn("'files':context['files']", roots)
        for action in plan:
            compile(action['response'][8:-4], '<action>', 'exec')
            self.assertNotIn("context['files'] =", action['response'])
            self.assertNotIn('test_visible.py', action['response'])
        self.assertEqual(sum(p['kind']=='subcall' for p in plan), 2)

    def test_learning_gate_requires_both_modes_and_no_losses(self):
        old = [dict(id='a', condition='ordinary', passed=False, recursive_passed=False, repl_passed=False),
               dict(id='b', condition='recursive', passed=False, recursive_passed=False, repl_passed=False)]
        new = copy.deepcopy(old)
        new[1].update(passed=True, recursive_passed=True)
        self.assertFalse(multifile.mode_gains(old,new)['passed'])
        new[0].update(passed=True,repl_passed=True)
        self.assertTrue(multifile.mode_gains(old,new)['passed'])
        old[0].update(passed=True,repl_passed=True)
        self.assertFalse(multifile.mode_gains(old,new)['passed'])
        with self.assertRaises(ValueError): multifile.mode_gains(old,new[::-1])

    def test_mode_gate_rejects_missing_test_evidence_and_duplicate_rows(self):
        old = [dict(id='a',condition='ordinary',passed=False,recursive_passed=False,repl_passed=False),
               dict(id='b',condition='recursive',passed=False,recursive_passed=False,repl_passed=False)]
        new = [dict(old[0],passed=True,repl_passed=True),
               dict(old[1],passed=True,repl_passed=True,recursive_passed=True)]
        self.assertTrue(multifile.mode_gains(old,new)['passed'])
        for field in ('repl_passed','recursive_passed','passed'):
            bad=copy.deepcopy(new);del bad[0][field]
            with self.subTest(field=field),self.assertRaises(ValueError):
                multifile.mode_gains(old,bad)
        with self.assertRaises(ValueError):
            multifile.mode_gains(old+[old[0]],new+[new[0]])
        bad=copy.deepcopy(new);bad[0]['condition']='unknown'
        with self.assertRaises(ValueError):
            multifile.mode_gains(bad,bad)


if __name__ == '__main__': unittest.main()
