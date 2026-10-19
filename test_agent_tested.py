"""The child must execute public checks; private grading never enters its prompt."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import agent_curriculum as curriculum
import recursive_agent as agent
import teacher_data
import agent_tested as tested
from test_agent_curriculum import LocalSandbox, REVISION
from test_rlm_tasks import run


def fixture():
    family=next('increment_'+str(i) for i in range(20)
                if teacher_data.split_for({'family':'increment_'+str(i)})=='train')
    return dict(id='fixture',family=family,prompt='Return x plus one.',
        answer=f'def {family}(x):\n    return x+1\n',
        tests='\n'.join(f'assert {family}({n}) == {n+1}' for n in (0,1,77,101,203)),
        provenance=dict(dataset='test/functions',revision=REVISION,source_row_sha256='a'*64))


class TaskTests(unittest.TestCase):
    def test_public_and_private_checks_are_disjoint(self):
        entry=fixture();task=tested.task(entry)
        self.assertEqual(task['split'],'train')
        self.assertEqual(task['editable'],['solver.py'])
        self.assertIn('(0)',task['files']['test_visible.py'])
        self.assertNotIn('(77)',str(task['files']))
        self.assertIn('(77)',task['checks'])
        self.assertNotIn('(0)',task['checks'])
        self.assertNotEqual(task['files']['solver.py'],entry['answer'])
        self.assertEqual(task['reference_patch'],{'solver.py':entry['answer']})
        self.assertNotEqual(run(task['files'],task['checks']).returncode,0)
        self.assertEqual(run({**task['files'],**task['reference_patch']},task['checks']).returncode,0)

    def test_reserved_families_and_near_copies_are_excluded(self):
        entry=fixture();task=tested.task(entry)
        self.assertFalse(tested.eligible(entry,[dict(task,repository=task['repository'])]))
        changed=copy.deepcopy(entry)
        changed['prompt']=' '.join('unique'+str(i) for i in range(20))
        self.assertFalse(tested.eligible(changed,[dict(task,repository='unrelated',prompt=changed['prompt'])]))
        self.assertTrue(tested.eligible(entry,[dict(task,repository='unrelated',prompt='Different task',files={})]))
        with patch.object(teacher_data,'split_for',return_value='dev'),self.assertRaises(ValueError):tested.task(entry)

    def test_plan_keeps_reference_out_of_parent_and_tests_in_child(self):
        task=tested.task(fixture());plan=tested.actions(task,True)
        self.assertEqual(len(plan),5)
        child=[p for p in plan if p['kind']=='subcall']
        self.assertEqual(len(child),2)
        self.assertTrue(all('subprocess.run' in p['response'] for p in child))
        roots='\n'.join(p['response'] for p in plan if p['kind']=='root')
        self.assertNotIn(repr(task['reference_patch']),roots)
        self.assertIn('json.loads(child_reply)',roots)
        self.assertNotIn(task['checks'],str(plan))
        for p in plan:compile(p['response'][8:-4],'<action>','exec')
        source_plan=tested.actions(task,True,visible=False)
        self.assertIn("context['files']['test_visible.py'] =",source_plan[0]['response'])
        self.assertNotIn(task['checks'],str(source_plan))

    def test_student_start_requires_both_visible_and_private_failure(self):
        task=tested.task(fixture())
        for private,public,expected in ((False,False,True),(False,True,False),(True,False,False)):
            with patch.object(agent,'grade',side_effect=[dict(passed=private),dict(passed=public)]):
                self.assertEqual(tested.failed_start(task),expected)

    def test_preflight_rejects_incomplete_collection_and_unexecuted_targets(self):
        from test_qwen35_training import Tokenizer
        from qwen35_train import sha256
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'traces').mkdir()
            with self.assertRaisesRegex(ValueError,'complete collection'):
                tested.finalize(folder,REVISION,folder/'unused',Tokenizer())
            call=dict(messages=[dict(role='user',content='task')],response='```repl\npass\n```',kind='root')
            record=dict(result=dict(trace=[call],calls=1),rows=[dict(messages=[],origin='root')])
            path=folder/'traces/one.json';path.write_text(json.dumps(record))
            (folder/'collection.json').write_text(json.dumps(dict(completed=True,traces={'one.json':sha256(path)})))
            with self.assertRaisesRegex(ValueError,'executed trace'):
                tested.finalize(folder,REVISION,folder/'unused',Tokenizer())
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):
                tested.finalize(folder,REVISION,folder/'unused',Tokenizer())
            excluded=dict(excluded_reason='Cell output limit exceeded',rows=[],result=dict(
                trace=[call],calls=1,passed=False,error='RuntimeError: Cell output limit exceeded',infrastructure_errors=[]))
            path.write_text(json.dumps(excluded))
            (folder/'collection.json').write_text(json.dumps(dict(completed=True,traces={'one.json':sha256(path)})))
            with self.assertRaisesRegex(ValueError,'Fewer than 64'):
                tested.finalize(folder,REVISION,folder/'unused',Tokenizer())
            excluded['result']['infrastructure_errors']=['Docker failed']
            path.write_text(json.dumps(excluded))
            (folder/'collection.json').write_text(json.dumps(dict(completed=True,traces={'one.json':sha256(path)})))
            with self.assertRaisesRegex(ValueError,'Invalid output-limit'):
                tested.finalize(folder,REVISION,folder/'unused',Tokenizer())

    @unittest.skipUnless(importlib.util.find_spec('rlm'),'Pinned upstream environment required')
    def test_output_limit_preserves_failed_evidence_and_does_not_hide_infrastructure_errors(self):
        task=tested.task(fixture())
        result=dict(error='RuntimeError: Cell output limit exceeded',infrastructure_errors=[],trace=[],passed=False)
        with patch.object(agent,'grade',return_value=dict(passed=False)),patch.object(agent,'solve',return_value=result):
            with self.assertRaises(curriculum.TrajectoryOutputLimit) as failure:
                curriculum.trajectory(task,False,True,REVISION,plan=[])
            self.assertIs(failure.exception.result,result)
            self.assertEqual(failure.exception.executions,[])
            result['infrastructure_errors']=['Docker failed']
            with self.assertRaises(ValueError) as failure:
                curriculum.trajectory(task,False,True,REVISION,plan=[])
            self.assertNotIsInstance(failure.exception,curriculum.TrajectoryOutputLimit)

    @unittest.skipUnless(importlib.util.find_spec('rlm'),'Pinned upstream environment required')
    def test_child_tests_execute_before_exact_patch_is_applied(self):
        task=tested.task(fixture())
        def grade(t,changes):return dict(passed=run({**t['files'],**changes},t['checks']).returncode==0)
        for recursive in (False,True):
            with warnings.catch_warnings(),patch.object(agent,'Sandbox',LocalSandbox),patch.object(agent,'grade',grade):
                warnings.simplefilter('ignore',ResourceWarning)
                record=curriculum.trajectory(task,recursive,True,REVISION,
                    plan=tested.actions(task,recursive),instruction=tested.INSTRUCTION if recursive else '')
            self.assertTrue(record['verification']['final_patch_passed'])
            if recursive:
                self.assertTrue(tested.recursive_success(record['result'],task))
                query=record['result']['queries'][0]
                child=record['executions'][query['execution_start']:query['execution_end']]
                self.assertIn('Public test exit: 1',child[0]['stdout'])
                self.assertIn('Public test exit: 0',child[1]['stdout'])


if __name__=='__main__':unittest.main()
