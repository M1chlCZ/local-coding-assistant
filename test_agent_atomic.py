import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import agent_atomic as atomic
from agent_multifile import task, mode_gains
from agent_pilot import evaluate_one, recursive_success, tested_repl
from repl_actions import Workspace
from test_agent_tested import fixture


class AtomicTests(unittest.TestCase):
    def test_every_recovery_executes_and_masks_only_failed_proposals(self):
        value=task(fixture(),0)
        for recovery in ('syntax','semantic','missing_edit'):
            with self.subTest(recovery=recovery),tempfile.TemporaryDirectory() as directory:
                answer={}
                workspace=Workspace(dict(task=value['prompt'],files=value['files'],editable=value['editable']),
                                    answer,root=directory)
                state=workspace.bindings()
                plan=atomic.actions(value,False,recovery)
                for action in plan:
                    with contextlib.redirect_stdout(io.StringIO()):
                        try:
                            exec(action['response'][8:-4],state,state)
                        except SyntaxError:
                            self.assertEqual(action.get('expected_error'),'SyntaxError')
                            self.assertFalse(action['train'])
                        else:
                            self.assertNotIn('expected_error',action)
                    self.assertNotIn(value['checks'], action['response'])
                self.assertEqual(answer['content'],value['reference_patch'])
                self.assertTrue(answer['ready'])
                self.assertNotEqual(workspace.checks[0]['returncode'],0)
                self.assertEqual(workspace.checks[-1]['returncode'],0)
                recursive=atomic.actions(value,True,recovery)
                self.assertLessEqual(len(recursive),atomic.LIMITS['calls'])
                self.assertLessEqual(sum(a['kind']=='subcall' for a in recursive),8)
                roots='\n'.join(a['response'] for a in recursive if a['kind']=='root')
                self.assertNotIn(value['reference_patch'][value['editable'][0]],roots)
                self.assertIn('apply(proposal)',roots)

    def test_checks_remain_executable_without_a_remembered_check_variable(self):
        import ast
        value=task(fixture(),0)
        for recovery in ('syntax','semantic','missing_edit'):
            for recursive in (False,True):
                plan=atomic.actions(value,recursive,recovery)
                for action in plan:
                    tree=ast.parse(action['response'][8:-4])
                    for call in ast.walk(tree):
                        if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='check':
                            self.assertEqual(ast.literal_eval(call.args[0]),value['diagnostic'])
                self.assertNotEqual(plan[-1]['response'].strip(),'```repl\nfinish()\n```')

    def test_frozen_interface_and_limits_reach_evaluator(self):
        row=dict(passed=False,infrastructure_errors=[])
        with patch('recursive_agent.solve',return_value=row) as solve, \
             patch('polyglot_runtime.check_program',return_value={'passed':True}):
            evaluate_one(dict(task={},condition='ordinary',strict_ordinary=True,
                              action_api=True,limits=atomic.LIMITS))
        self.assertTrue(solve.call_args.kwargs['action_api'])
        self.assertEqual(solve.call_args.kwargs['calls'],12)
        self.assertEqual(solve.call_args.kwargs['mode'],'baseline')

    def test_child_answer_alone_cannot_count_as_tested_recursion(self):
        value=task(fixture(),0)
        proposal=value['reference_patch']
        hashes={p:hashlib.sha256(s.encode()).hexdigest() for p,s in proposal.items()}
        final=json.dumps(proposal)
        checks=[dict(returncode=1,assertions=1,before={},after={}),
                dict(returncode=0,assertions=1,before=hashes,after=hashes)]
        execution=dict(stderr='',final_answer=final,checks=checks)
        row=dict(passed=True,patch=proposal,workspace_hashes=hashes,action_api=True,
            executions=[execution],queries=[dict(recursive=True,response=final,execution_start=0,execution_end=1)])
        self.assertFalse(recursive_success(row,value))
        self.assertFalse(tested_repl(row), 'Child checks cannot stand in for parent checks')
        row['executions'].append(dict(stderr='',final_answer=final,checks=[checks[-1]]))
        self.assertTrue(tested_repl(row))
        self.assertTrue(recursive_success(row,value))
        for bad in ([],checks[:1],[dict(checks[-1],assertions=0)],
                    [checks[0],dict(checks[-1],after={})]):
            execution['checks']=bad
            self.assertFalse(recursive_success(row,value))

    def test_ordinary_patch_without_execution_is_not_a_repl_gain(self):
        before=[dict(id='a',condition='ordinary',passed=False,recursive_passed=False,repl_passed=False),
                dict(id='b',condition='recursive',passed=False,recursive_passed=False,repl_passed=False)]
        after=[dict(before[0],passed=True),dict(before[1],passed=True,recursive_passed=True)]
        self.assertFalse(mode_gains(before,after)['passed'])
        after[0]['repl_passed']=True
        self.assertTrue(mode_gains(before,after)['passed'])


if __name__=='__main__':
    unittest.main()
