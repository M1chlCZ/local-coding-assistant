"""A printed subcall or partial comparison cannot unlock the pilot gate."""
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from agent_pilot import recursive_success, comparison, evaluate_one
from qwen35_train import atomic_json, sha256


class PilotTests(unittest.TestCase):
    def test_frozen_recursive_instruction_reaches_worker(self):
        entry=dict(condition='recursive',instruction='Execute the supplied public tests in the child.',
                   task=dict(id='repair',repository='repair',split='dev'))
        row=dict(id='repair',passed=False,error='',elapsed_s=1,infrastructure_errors=[])
        with patch('recursive_agent.solve',return_value=row) as solve, \
             patch('polyglot_runtime.check_program',return_value={'passed':True}):
            evaluate_one(entry)
        self.assertEqual(solve.call_args.kwargs['instruction'],entry['instruction'])

    @unittest.skipUnless(importlib.util.find_spec('rlm'),'Pinned upstream environment required')
    def test_invalid_model_answer_is_distinct_from_broken_protocol(self):
        import recursive_agent as agent
        from rlm.clients.openai import OpenAIClient
        from rlm.core.types import ModelUsageSummary
        task=dict(id='invalid-answer',repository='fixture',split='dev',prompt='Repair solver',
                  files={'solver.py':'x=1'},editable=['solver.py'])
        for frame,infrastructure in [
                (dict(type='result',stdout='',stderr='',final_answer=[]),False),
                (dict(type='result',stdout='',stderr='',final_answer='x'*100001),False),
                (dict(type='result',stdout=[],stderr='',final_answer=None),True),
                (dict(type='result',stdout='',stderr='',final_answer=None,pipe_error=True),True)]:
            class BoundarySandbox(agent.Sandbox):
                def __init__(self,context_payload,budget=None,depth=0,**unused):
                    self.budget,self.depth,self.editable=budget,depth,[]
                    self.workspace_hashes={}
                def _send(self,value):pass
                def _receive(self):
                    if frame.get('pipe_error'):raise BrokenPipeError('REPL transport unavailable')
                    return frame
                def cleanup(self):pass
            with self.subTest(infrastructure=infrastructure,answer_type=type(frame['final_answer']).__name__), \
                 patch.object(agent,'Sandbox',BoundarySandbox), \
                 patch.object(OpenAIClient,'completion',return_value="```repl\nanswer['ready'] = True\n```"), \
                 patch.object(OpenAIClient,'get_last_usage',return_value=ModelUsageSummary(1,1,1)):
                if infrastructure:
                    with self.assertRaisesRegex(RuntimeError,'Infrastructure failure:'):
                        agent.solve(task,'http://127.0.0.1:1',calls=4,seconds=10)
                    continue
                row=agent.solve(task,'http://127.0.0.1:1',calls=4,seconds=10)
            self.assertFalse(row['passed'])
            self.assertFalse(row.get('patch'))
            self.assertEqual(bool(row['infrastructure_errors']),infrastructure)
            if not infrastructure:
                self.assertIn('Invalid model final answer',row['error'])

    @unittest.skipUnless(importlib.util.find_spec('rlm'),'Pinned upstream environment required')
    def test_swallowed_outage_raises_after_cleanup_despite_final_patch(self):
        import recursive_agent as agent
        from test_agent_curriculum import LocalSandbox
        from rlm.clients.openai import OpenAIClient
        from rlm.core.types import ModelUsageSummary
        created=[];cleaned=[]
        class TrackedSandbox(LocalSandbox):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);created.append(self)
            def cleanup(self):
                super().cleanup();cleaned.append(id(self))
        responses=iter(["```repl\nchild = rlm_query('inspect selected source')\n"
                        "answer['content'] = {'solver.py':'x=2'}; answer['ready'] = True\n```"])
        def reply(*args,**kwargs):
            try:return next(responses)
            except StopIteration:raise ConnectionError('child service unavailable')
        task=dict(id='outage',repository='fixture',split='train',prompt='Repair solver',
                  files={'solver.py':'x=1'},editable=['solver.py'])
        with patch.object(agent,'Sandbox',TrackedSandbox), \
             patch.object(OpenAIClient,'completion',side_effect=reply), \
             patch.object(OpenAIClient,'get_last_usage',return_value=ModelUsageSummary(1,1,1)):
            with self.assertRaisesRegex(RuntimeError,'Infrastructure failure: ConnectionError: child service unavailable'):
                agent.solve(task,'http://127.0.0.1:1',depth=2,calls=4,seconds=10)
        self.assertEqual(len(created),2)
        self.assertTrue(all(id(env) in cleaned for env in created))
        self.assertTrue(any(env.state['answer']==dict(content={'solver.py':'x=2'},ready=True) for env in created))

    def test_training_requires_current_complete_trajectory_preflight(self):
        import agent_pilot as pilot
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); dataset=root/'train.jsonl'
            dataset.write_text('verified rows\n')
            manifest=Path(str(dataset)+'.manifest.json')
            atomic_json(manifest,dict(rows=550,dataset_sha256=sha256(dataset)))
            revision='a'*40
            evidence=dict(passed=True,all_agent_actions_included=True,complete_trajectories=True,
                dataset_sha256=sha256(dataset),manifest_sha256=sha256(manifest),
                tokenizer_model='Qwen/Qwen3.5-4B',tokenizer_revision=revision)
            atomic_json(root/'feasibility.json',evidence)
            self.assertEqual(pilot.verified_curriculum(dataset,revision),550)
            for key,value in [('passed',False),('all_agent_actions_included',False),
                              ('complete_trajectories','true'),('dataset_sha256','0'*64),
                              ('manifest_sha256','0'*64),('tokenizer_revision','b'*40)]:
                atomic_json(root/'feasibility.json',dict(evidence,**{key:value}))
                with self.subTest(field=key),self.assertRaises(ValueError):
                    pilot.verified_curriculum(dataset,revision)
            atomic_json(root/'feasibility.json',evidence)
            dataset.write_text('changed verified rows\n')
            with self.assertRaises(ValueError):pilot.verified_curriculum(dataset,revision)
            (root/'feasibility.json').unlink()
            with self.assertRaises(ValueError):pilot.verified_curriculum(dataset,revision)

    def test_grader_infrastructure_errors_are_not_scored(self):
        import recursive_agent as agent
        task=dict(files={'solver.py':'x=0'},editable=['solver.py'],checks='assert True')
        for code in (125,126,127):
            with self.subTest(exit_code=code), \
                 patch.object(agent,'bounded_run',return_value=dict(exit_code=code,output_limited=False,
                             output='docker runtime unavailable')), patch.object(agent.subprocess,'run'):
                with self.assertRaisesRegex(RuntimeError,'Grader container infrastructure failed'):
                    agent.grade(task,{'solver.py':'x=1'})
        with patch.object(agent,'bounded_run',return_value=dict(exit_code=1,output_limited=False,
                          output='AssertionError: incorrect result')), patch.object(agent.subprocess,'run'):
            self.assertFalse(agent.grade(task,{'solver.py':'x=1'})['passed'])

    def test_swallowed_child_infrastructure_error_is_not_scored(self):
        entry=dict(condition='recursive',task=dict(id='repair',repository='repair',split='dev'))
        for error in ('', 'RuntimeError: Model call or output-token budget exhausted'):
            with self.subTest(top_level_error=error):
                row=dict(id='repair',passed=False,error=error,elapsed_s=10,
                         infrastructure_errors=['InternalServerError: child API returned 503'])
                with patch('recursive_agent.solve',return_value=row), \
                     patch('polyglot_runtime.check_program',return_value={'passed':True}):
                    with self.assertRaisesRegex(RuntimeError,'Infrastructure failure'):
                        evaluate_one(entry)

    def test_model_budget_exhaustion_remains_scored_failure(self):
        entry=dict(condition='recursive',task=dict(id='repair',repository='repair',split='dev'))
        row=dict(id='repair',passed=False,elapsed_s=240,infrastructure_errors=[],
                 error='RuntimeError: Model call or output-token budget exhausted')
        with patch('recursive_agent.solve',return_value=row), \
             patch('polyglot_runtime.check_program',return_value={'passed':True}):
            result=evaluate_one(entry)
        self.assertFalse(result['passed'])
        self.assertFalse(result['recursive_passed'])

    def test_recursive_application(self):
        task=dict(files={'solver.py':'x=1'},editable=['solver.py'])
        patch={'solver.py':'x=2'}; answer=json.dumps(patch)
        row=dict(passed=True,patch=patch,workspace_hashes={'solver.py':hashlib.sha256(b'x=2').hexdigest()},
                 queries=[dict(recursive=True,response=answer,execution_start=0,execution_end=1)],
                 executions=[dict(stderr='',final_answer=answer)])
        self.assertTrue(recursive_success(row,task))
        for key,value in [('queries',[]),('executions',[]),('workspace_hashes',{}),('passed',False)]:
            bad=copy.deepcopy(row);bad[key]=value
            self.assertFalse(recursive_success(bad,task))
        row['queries'][0]['recursive']=False
        self.assertFalse(recursive_success(row,task))

    def test_matched_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            entries=[dict(condition='recursive' if i else 'ordinary',task=dict(id=str(i),repository=str(i))) for i in range(3)]
            revision='a'*40
            atomic_json(root/'evaluation.json',dict(model_revision=revision,development=entries,retention=[dict(id='anchor')]))
            adapter=root/'candidate/adapter';adapter.mkdir(parents=True)
            (adapter/'adapter_model.safetensors').write_bytes(b'candidate weights')
            atomic_json(adapter/'adapter_config.json',dict(r=8))
            atomic_json(adapter/'training.json',dict(status='completed',model='Qwen/Qwen3.5-4B',
                revision=revision,precision='bf16',adapter_sha256=sha256(adapter/'adapter_model.safetensors'),
                adapter_config_sha256=sha256(adapter/'adapter_config.json')))
            binding=dict(evaluation_sha256=sha256(root/'evaluation.json'),suite='development')
            before=[dict(id=str(i),condition=e['condition'],passed=i==2,recursive_passed=False) for i,e in enumerate(entries)]
            after=[dict(**{**r,'passed':True,'recursive_passed':i==1}) for i,r in enumerate(before)]
            def save(name,rows,retained=True):
                evidence=dict(binding,adapter_sha256=sha256(adapter/'adapter_model.safetensors') if name=='candidate' else None)
                atomic_json(root/name/'development.json',dict(completed=True,binding=evidence,rows=rows,
                            retention=[dict(id='anchor',passed=retained)]))
            save('baseline',before);save('candidate',after)
            result=comparison(root,'candidate','development')
            self.assertTrue(result['pilot_passed']);self.assertFalse(result['full_loop_authorized'])
            self.assertEqual(result['inputs']['candidate/development.json'],sha256(root/'candidate/development.json'))
            for name,key,value in [('baseline','adapter_sha256','b'*64),('candidate','adapter_sha256',None),
                                   ('candidate','suite','holdout')]:
                path=root/name/'development.json';report=json.loads(path.read_text())
                report['binding'][key]=value;atomic_json(path,report)
                with self.subTest(report=name,key=key), self.assertRaises(ValueError):
                    comparison(root,'candidate','development')
                save(name,before if name=='baseline' else after)
            original_weights=(adapter/'adapter_model.safetensors').read_bytes()
            (adapter/'adapter_model.safetensors').write_bytes(b'replaced weights')
            with self.assertRaises(ValueError):comparison(root,'candidate','development')
            (adapter/'adapter_model.safetensors').write_bytes(original_weights)
            report=json.loads((root/'candidate/development.json').read_text())
            report['rows'][1]['recursive_passed']='true';atomic_json(root/'candidate/development.json',report)
            with self.assertRaises(ValueError):comparison(root,'candidate','development')
            save('candidate',after)
            save('candidate',after,False)
            self.assertFalse(comparison(root,'candidate','development')['pilot_passed'])

            save('candidate',after[:-1])
            with self.assertRaises(ValueError):comparison(root,'candidate','development')
            save('candidate',after)
            entries[1]['task']['repository']='0'
            atomic_json(root/'evaluation.json',dict(model_revision=revision,development=entries,retention=[dict(id='anchor')]))
            binding['evaluation_sha256']=sha256(root/'evaluation.json')
            save('baseline',before);save('candidate',after)
            self.assertFalse(comparison(root,'candidate','development')['pilot_passed'])

            config=json.loads((root/'evaluation.json').read_text())
            config['baseline_adapter_sha256']='b'*64
            atomic_json(root/'evaluation.json',config)
            binding['evaluation_sha256']=sha256(root/'evaluation.json')
            save('baseline',before);save('candidate',after)
            with self.assertRaises(ValueError):comparison(root,'candidate','development')
            report=json.loads((root/'baseline/development.json').read_text())
            report['binding']['adapter_sha256']='b'*64
            atomic_json(root/'baseline/development.json',report)
            self.assertFalse(comparison(root,'candidate','development')['pilot_passed'])


if __name__=='__main__':unittest.main()
