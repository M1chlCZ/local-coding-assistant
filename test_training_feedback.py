"""Training-only correction, durable retries, and positive-target separation."""
import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path

from train_adapter import atomic_json,encode_row
from training_data import export,load_verified,sha256


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.attempt=self.root/'attempt.json'
        self.task={'id':'training-r2','repository':'training','split':'train','prompt':'Repair solution.py; run visible tests.',
            'files':{'solution.py':'broken','test_visible.py':'visible'},'editable':['solution.py'],'checks':'HIDDEN_CHECK_CANARY'}
        self.registry=self.root/'tasks.json';atomic_json(self.registry,[self.task])
        self.settings={'mode':'rlm','depth':2,'instruction':''};self.limits={'calls':8,'output_tokens':4096,'seconds':180}
        self.calls=[];self.outcomes=['broken','correct'];self.pause=False
        self.module=importlib.import_module('training_feedback')

    def solve(self,task,base,**limits):
        self.calls.append((copy.deepcopy(task),limits))
        answer=self.outcomes.pop(0)
        trace={'kind':'root','messages':[{'role':'user','content':task['prompt']}],
            'response':answer,'input_tokens':10,'output_tokens':2}
        return {'id':task['id'],'repository':task['repository'],'split':task['split'],'mode':'rlm',
            'trace':[trace],'patch':{'solution.py':answer},'passed':False,'calls':2,'elapsed_s':3}

    def grade(self,task,patch):
        return {'passed':patch['solution.py']=='correct','feedback':'HIDDEN_FEEDBACK_CANARY'}

    def collect(self,**changes):
        args={'solve':self.solve,'grade':self.grade,'settings':self.settings,'limits':self.limits,
              'interrupted':lambda:self.pause,'remaining_seconds':lambda:999,'before_attempt':lambda n:None}
        args.update(changes)
        return self.module.collect(self.task,'http://local',self.attempt,sha256(self.registry),**args)

    def test_failed_attempt_is_corrected_and_only_correction_becomes_a_target(self):
        row=self.collect()
        self.assertTrue(row['passed']);self.assertEqual(len(self.calls),2)
        retry_prompt=self.calls[1][0]['prompt']
        self.assertIn('previous',retry_prompt.lower());self.assertIn('broken',retry_prompt)
        self.assertNotIn('HIDDEN_FEEDBACK_CANARY',retry_prompt);self.assertNotIn('HIDDEN_CHECK_CANARY',retry_prompt)
        self.assertEqual(row['retry_info']['attempts'],2);self.assertTrue(row['retry_info']['recovered'])
        self.assertEqual(len(json.loads(self.attempt.read_text())['attempts']),2)
        report=self.root/'teacher.json';atomic_json(report,{'schema_version':1,'split':'train',
            'tasks_sha256':sha256(self.registry),'tasks':[row]})
        dataset=self.root/'training.jsonl';export([report],self.registry,dataset)
        rows=load_verified(dataset,self.registry)
        self.assertEqual([r['messages'][-1]['content'] for r in rows],['correct'])
        self.assertEqual(self.collect()['trace'],row['trace']);self.assertEqual(len(self.calls),2)

    def test_first_success_skips_retry_and_two_failures_are_excluded(self):
        self.outcomes=['correct'];self.assertTrue(self.collect()['passed']);self.assertEqual(len(self.calls),1)
        self.attempt.unlink();self.outcomes=['broken','still broken'];self.calls=[]
        row=self.collect();self.assertFalse(row['passed']);self.assertEqual(len(self.calls),2)
        self.assertFalse(row['retry_info']['recovered'])
        report=self.root/'teacher.json';atomic_json(report,{'schema_version':1,'split':'train',
            'tasks_sha256':sha256(self.registry),'tasks':[row]})
        with self.assertRaisesRegex(ValueError,'No successful'):export([report],self.registry,self.root/'failed.jsonl')

    def test_pause_after_failure_resumes_second_attempt_without_repeating_first(self):
        def grade(task,patch):
            self.pause=True
            return self.grade(task,patch)
        self.assertIsNone(self.collect(grade=grade));self.assertEqual(len(self.calls),1)
        self.pause=False
        self.assertTrue(self.collect()['passed']);self.assertEqual(len(self.calls),2)

    def test_grade_interruption_reuses_saved_model_response(self):
        self.outcomes=['correct']
        def broken_grader(task,patch):raise RuntimeError('Docker temporarily unavailable')
        with self.assertRaises(RuntimeError):self.collect(grade=broken_grader)
        self.assertEqual(len(self.calls),1)
        self.assertTrue(self.collect()['passed']);self.assertEqual(len(self.calls),1)

    def test_pause_and_exhausted_clock_prevent_generation(self):
        self.pause=True;self.assertIsNone(self.collect());self.assertEqual(self.calls,[])
        self.pause=False;self.assertIsNone(self.collect(remaining_seconds=lambda:0));self.assertEqual(self.calls,[])
        self.outcomes=['correct'];self.collect(remaining_seconds=lambda:12)
        self.assertEqual(self.calls[0][1]['seconds'],12)

    def test_evaluation_splits_and_registry_changes_cannot_be_retried(self):
        for split in ('dev','holdout','custom'):
            self.task['split']=split
            with self.assertRaises(ValueError):self.collect()
        self.assertEqual(self.calls,[])
        self.task['split']='train';self.outcomes=['correct'];self.collect()
        self.task['prompt']='Changed task'
        with self.assertRaisesRegex(ValueError,'binding'):self.collect()

    def test_visible_diagnostics_and_timeout_advice_are_bounded_and_private_checks_are_omitted(self):
        row={'passed':False,'error':'APITimeoutError: Request timed out.', 'feedback':'PRIVATE_CANARY',
            'trace':[{'kind':'root','messages':[{'role':'user','content':'REPL output:\nCompile failure: visible compiler error'}]}]}
        message=self.module.feedback(self.task,row)
        self.assertIn('time',message.lower());self.assertIn('visible compiler error',message)
        self.assertNotIn('PRIVATE_CANARY',message);self.assertLessEqual(len(message),4096)
        row['trace'][0]['messages'][0]['content']='REPL output:\n'+'x'*100000
        self.assertLessEqual(len(self.module.feedback(self.task,row)),4096)

    def test_non_model_infrastructure_failure_is_not_taught_as_a_correction(self):
        def offline(task,base,**limits):
            row={'id':task['id'],'repository':task['repository'],'split':'train','passed':False,
                 'mode':'rlm','trace':[],'error':'APIConnectionError: Connection error.'}
            self.calls.append(task);return row
        with self.assertRaisesRegex(RuntimeError,'infrastructure'):self.collect(solve=offline)
        self.assertEqual(len(self.calls),1)


if __name__=='__main__':unittest.main()
