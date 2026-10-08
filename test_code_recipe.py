"""Checks for balanced source targets, fresh-base candidates and held-out isolation."""
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from train_adapter import atomic_json, parser
from training_data import load_verified, sha256
import learning_session as learning


class CodeRecipeTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('code_recipe'), 'Direct coding recipe is missing')
        return importlib.import_module('code_recipe')

    def task(self, language='python', index=0, split='train'):
        filename={'python':'solution.py','go':'solution.go','typescript':'solution.ts',
                  'rust':'solution.rs','dart':'solution.dart'}[language]
        return {'id':f'{language}-{index}', 'repository':f'problem-{index}', 'split':split,
            'language':language, 'filename':filename, 'editable':[filename],
            'prompt':f'Add two integers (task {index}).\nInspect and edit the source through the Python REPL. Run test_visible.py.',
            'files':{filename:'broken', 'test_visible.py':"cases=[{'input':'2 3','output':'5'}]"},
            'cases':[{'input':'SECRET INPUT','output':'SECRET OUTPUT'}],
            'reference_patch':{filename:'SECRET REFERENCE'}}

    def test_prompt_contains_only_visible_source_and_examples(self):
        task=self.task();messages=self.module().messages(task)
        text=json.dumps(messages)
        self.assertIn('2 3',text);self.assertIn('complete source',text)
        self.assertNotIn('SECRET',text);self.assertNotIn('through the Python REPL',text)

    def test_direct_source_answer_is_not_a_repl_action(self):
        module=self.module()
        self.assertEqual(module.source('```go\npackage main\nfunc main() {}\n```'), 'package main\nfunc main() {}')
        with self.assertRaises(ValueError):module.source('```repl\nprint(context)\n```')

    def test_backticks_inside_valid_source_do_not_change_the_target(self):
        code="text = '''```python\nprint(1)\n```'''\nprint(text)"
        self.assertEqual(self.module().source(code),code)

    def test_balanced_export_regrades_and_uses_one_complete_source_per_task(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tasks=[self.task(l,i) for l in module.LANGUAGES for i in range(3 if l=='python' else 1)]
            registry=root/'tasks.json';atomic_json(registry,tasks)
            report={'schema_version':1,'split':'train','tasks_sha256':sha256(registry),'tasks':[
                {**{k:t[k] for k in ('id','repository','split')},'mode':'direct','passed':True,
                 'patch':{t['filename']:"text = '''```python\nprint(1)\n```'''\nprint(text)" if t['language']=='python'
                          else 'valid '+t['language']},'trace':[{'response':'BAD REPL TARGET'}]} for t in tasks]}
            original=root/'teacher.json';atomic_json(original,report);calls=[]
            def grade(task,patch):calls.append(task['id']);return {'passed':True}
            with patch.object(module,'grade',side_effect=grade):
                result=module.export_code([original],registry,root/'training.jsonl')
            rows=load_verified(root/'training.jsonl',registry)
            self.assertEqual(len(rows),7);self.assertEqual(result['languages']['python'],3)
            self.assertEqual(result['languages']['go'],1)
            self.assertEqual(set(calls),{t['id'] for t in tasks})
            self.assertTrue(all(len(r['messages'])==3 for r in rows))
            self.assertEqual(rows[0]['messages'][-1]['content'],report['tasks'][0]['patch']['solution.py'])
            self.assertNotIn('BAD REPL TARGET',(root/'training.jsonl').read_text())

    def test_failed_or_development_answers_cannot_become_targets(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tasks=[self.task(l) for l in module.LANGUAGES]+[self.task('python',9,'dev')]
            registry=root/'tasks.json';atomic_json(registry,tasks)
            rows=[{**{k:t[k] for k in ('id','repository','split')},'mode':'direct','passed':True,
                   'patch':{t['filename']:'valid'}} for t in tasks]
            rows[1]['passed']=False
            p=root/'teacher.json';atomic_json(p,{'schema_version':1,'split':'train','tasks_sha256':sha256(registry),'tasks':rows})
            with patch.object(module,'grade',return_value={'passed':True}):
                with self.assertRaises(module.UnbalancedData):module.export_code([p],registry,root/'training.jsonl')
            self.assertFalse((root/'training.jsonl').exists())

    def test_training_starts_from_base_and_never_warms_from_the_old_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            session=learning.Session(Path(directory)/'session',6,Path('/teacher.gguf'),research=True)
            self.addCleanup(session.lock.close);self.addCleanup(session.gpu_lock.close)
            session.state.update(recipe='balanced-code-v1',round=2,best_adapter='/old-adapter',research_adapter='/old-research')
            args=parser().parse_args(session.training_command(session.path/'round-002')[2:])
            self.assertIsNone(args.warm_start)
            self.assertEqual(args.learning_rate,1e-5);self.assertEqual(args.max_steps,20)
            self.assertEqual(args.max_epochs,1)

    def test_visible_retry_feedback_never_includes_hidden_grading_output(self):
        from training_feedback import feedback
        result=feedback(self.task(),{'mode':'direct','error':'','visible_feedback':'Visible compiler error',
            'feedback':'SECRET HIDDEN FAILURE','patch':{'solution.py':'bad source'}})
        self.assertIn('Visible compiler error',result);self.assertNotIn('SECRET',result)
        self.assertIn('complete source',result);self.assertNotIn('JSON patch',result)

    def test_balanced_order_uses_all_languages_in_every_full_round(self):
        module=self.module();tasks=[self.task(l,i) for l in module.LANGUAGES for i in range(9)]
        ordered=module.balanced_order(tasks)
        self.assertEqual(len(ordered),45);self.assertEqual(len({t['id'] for t in ordered}),45)
        for start in (0,20):
            self.assertEqual({l:sum(t['language']==l for t in ordered[start:start+20]) for l in module.LANGUAGES},
                             {l:4 for l in module.LANGUAGES})

    def test_quality_checks_use_direct_answers_only_for_the_new_recipe(self):
        self.assertTrue(hasattr(learning,'quality_solver'))
        self.assertEqual(learning.quality_solver({'recipe':'balanced-code-v1'}).__module__,'code_recipe')
        self.assertEqual(learning.quality_solver({}).__module__,'recursive_agent')

    def test_balanced_token_filter_does_not_reintroduce_python_bias(self):
        import train_adapter
        self.assertTrue(hasattr(train_adapter,'encode_dataset'))
        rows=[{'language':l,'fits':True} for l in self.module().LANGUAGES for _ in range(2)]
        rows.append({'language':'python','fits':True});rows[1]['fits']=False
        with patch.object(train_adapter,'encode_row',side_effect=lambda tokenizer,row,length: {'labels':[1]} if row['fits'] else None):
            encoded,counts=train_adapter.encode_dataset(None,rows,4096,balanced=True)
        self.assertEqual(len(encoded),10);self.assertEqual(counts['python'],2)
        self.assertEqual(sum(r['language_weight'] for r in encoded),10)

    def test_token_filter_keeps_all_examples_and_balances_language_loss(self):
        import train_adapter
        rows=[{'language':l} for l in self.module().LANGUAGES]
        rows += [{'language':'python'}]*4
        with patch.object(train_adapter,'encode_row',return_value={'labels':[1]}):
            encoded,counts=train_adapter.encode_dataset(None,rows,4096,balanced=True)
        self.assertEqual(len(encoded),9)
        self.assertEqual(counts['python'],5)
        self.assertEqual(counts['go'],1)
        self.assertAlmostEqual(encoded[0]['language_weight'],9/25)
        self.assertAlmostEqual(encoded[1]['language_weight'],9/5)

    def test_recipe_trials_use_the_same_data_and_fixed_accepted_warm_start(self):
        with tempfile.TemporaryDirectory() as directory:
            session=learning.Session(Path(directory)/'session',6,Path('/teacher.gguf'),research=True)
            self.addCleanup(session.lock.close);self.addCleanup(session.gpu_lock.close)
            session.state.update(recipe='balanced-code-v1',recipe_trials=True,round=2,
                best_adapter='/new-accepted',research_adapter='/experimental',trial_origin_adapter='/accepted')
            for index,steps,warm in ((0,20,None),(1,500,None),(2,500,Path('/accepted'))):
                session.state['trial_index']=index
                args=parser().parse_args(session.training_command(session.path/'round-002')[2:])
                self.assertEqual(args.max_steps,steps)
                self.assertEqual(args.warm_start,warm)
                self.assertEqual(args.dataset,session.path/'round-002/training.jsonl')
                self.assertEqual(args.max_epochs,1)
                self.assertEqual(args.learning_rate,1e-5)
            self.assertEqual(len({str(session.training_output(session.path/'round-002',i)) for i in range(3)}),3)

    def test_replay_anchor_preserves_verified_examples_from_every_round(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory)/'parent';anchor=Path(directory)/'next/round-001'
            tasks=[self.task(l,i) for i in range(2) for l in module.LANGUAGES]
            for index in (1,2):
                folder=parent/f'round-{index:03d}'
                atomic_json(folder/'tasks.json',tasks[:index*5])
                atomic_json(folder/'teacher.json',{'schema_version':1,'split':'train',
                    'tasks_sha256':sha256(folder/'tasks.json'),'tasks':[
                        {**{k:t[k] for k in ('id','repository','split')},'mode':'direct','passed':True,
                         'patch':{t['filename']:'valid'}} for t in tasks[(index-1)*5:index*5]]})
            module.replay_anchor([parent],anchor)
            value=json.loads((anchor/'teacher.json').read_text())
            self.assertEqual(len(value['tasks']),10)
            self.assertEqual(value['tasks_sha256'],sha256(anchor/'tasks.json'))
            self.assertEqual(len(value['replay_sources']),2)
            with patch.object(module,'grade',return_value={'passed':True}):
                module.export_code([anchor/'teacher.json'],anchor/'tasks.json',anchor/'training.jsonl')
            self.assertEqual(len(load_verified(anchor/'training.jsonl',anchor/'tasks.json')),10)

    def test_all_three_recipe_trials_finish_before_collecting_new_data(self):
        from train_adapter import CHECKPOINT_FILES,complete_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            session=learning.Session(Path(directory)/'session',6,Path('/teacher.gguf'),research=True)
            self.addCleanup(session.lock.close);self.addCleanup(session.gpu_lock.close)
            tasks=[self.task('python',i,'dev') for i in range(3)]
            base={'tasks':[{'id':t['id'],'passed':False} for t in tasks]}
            origin={'tasks':[{'id':t['id'],'passed':i==0} for i,t in enumerate(tasks)]}
            session.state.update(recipe='balanced-code-v1',recipe_trials=True,fast_reject=True,round=2,
                best_adapter='/origin',best_dev=origin,trial_origin_adapter='/origin',trial_origin_dev=origin)
            folder=session.path/'round-002';folder.mkdir()
            atomic_json(session.path/'dev-base.json',base)
            for index in range(3):
                output=session.training_output(folder,index);checkpoint=output/'checkpoint-1';checkpoint.mkdir(parents=True)
                binding={'dataset_sha256':'a'*64,'warm_start_sha256':'b'*64 if index==2 else None}
                atomic_json(output/'run.json',binding)
                for name in CHECKPOINT_FILES:(checkpoint/name).write_text('fixture')
                complete_checkpoint(checkpoint,binding)
                successes=({0,1},{0,2},{0,1,2})[index]
                rows=[{'id':t['id'],'passed':i in successes} for i,t in enumerate(tasks)]
                with patch.object(session,'server',return_value=True),patch.object(session,'stop_child'), \
                     patch.object(session,'interrupted',return_value=False),patch.object(learning,'development_tasks',return_value=tasks), \
                     patch.object(learning,'quality_rows',side_effect=lambda state,chunk,url:[r for r in rows if r['id'] in {t['id'] for t in chunk}]),patch('evaluation.request'):
                    session.evaluate(folder)
                self.assertEqual(session.state['trial_origin_adapter'],'/origin')
                self.assertEqual(session.state['round'],2 if index<2 else 3)
                self.assertEqual(session.state['phase'],'train' if index<2 else 'collect')
                result=json.loads((folder/('result-'+output.name+'.json')).read_text())
                self.assertTrue(result['complete'])
                self.assertEqual(result['accepted'],index!=1)
            self.assertEqual(len(list(folder.glob('result-adapter-*.json'))),3)

    def test_replay_suffix_does_not_turn_old_answers_into_fresh_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            session=learning.Session(Path(directory)/'session',6,Path('/teacher.gguf'),research=True)
            self.addCleanup(session.lock.close);self.addCleanup(session.gpu_lock.close)
            session.state.update(recipe='balanced-code-v1',recipe_trials=True,round=2,polyglot_baseline_complete=True)
            old={**self.task('go',0),'id':'old-go-r2','source_reference_passed':True}
            new={**self.task('go',1),'id':'new-go-r2','source_reference_passed':True}
            atomic_json(session.path/'round-001/tasks.json',[old])
            folder=session.path/'round-002';atomic_json(folder/'tasks.json',[old,new]);seen=[]
            def collect(task,*args,**kwargs):
                seen.append(task['id'])
                return {**{k:task[k] for k in ('id','repository','split')},'passed':True,'mode':'direct'}
            with patch.object(session,'server',return_value=True),patch.object(session,'interrupted',return_value=False), \
                 patch('training_feedback.collect',side_effect=collect):
                session.collect(folder)
            self.assertEqual(seen,['new-go-r2'])

    def test_zero_delta_baseline_is_measured_once_with_direct_checks(self):
        from polyglot_session import establish_baseline
        with tempfile.TemporaryDirectory() as directory:
            session=learning.Session(Path(directory)/'session',6,Path('/teacher.gguf'),research=True)
            self.addCleanup(session.lock.close);self.addCleanup(session.gpu_lock.close)
            adapter=session.path/'zero';adapter.mkdir();(adapter/'adapter_model.safetensors').write_text('zero')
            session.state.update(recipe='balanced-code-v1',baseline_equivalent=True,best_adapter=str(adapter),best_dev={'tasks':[]},confirmation={})
            tasks=[self.task('python',0,'dev'),self.task('go',0,'dev')]
            atomic_json(session.path/'development-tasks.json',tasks);calls=[]
            def solve(task,base,**settings):
                self.assertEqual(settings['mode'],'direct','New baseline still uses the old RLM protocol')
                calls.append(settings['mode']);return {'id':task['id'],'passed':True,'patch':{task['filename']:'valid'}}
            with patch.object(session,'server',return_value=True),patch.object(session,'stop_child'), \
                 patch.object(session,'interrupted',return_value=False),patch.object(learning,'development_tasks',return_value=tasks), \
                 patch.object(learning,'quality_solver',return_value=solve),patch('evaluation.request'), \
                 patch('recursive_agent.grade',return_value={'passed':True}),patch('recursive_agent.solve',side_effect=solve):
                self.assertTrue(establish_baseline(session,session.path))
            self.assertEqual(calls,['direct','direct'])
            self.assertEqual(session.state['research_score'],2)
            report=json.loads((session.path/'dev-base.json').read_text())
            self.assertEqual(report['settings']['mode'],'direct')

    def test_recipe_preparation_never_overrides_a_running_or_stopped_worker(self):
        module=self.module()
        self.assertTrue(hasattr(module,'prepare_recipe'))
        from continuous_learning import Controller
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);child=root/'child'
            atomic_json(child/'status.json',{'status':'running','active_seconds':123})
            atomic_json(child/'command.json',{'action':'resume'})
            controller=Controller.create(root/'controller',child)
            for action in ('resume','stop'):
                atomic_json(controller.path/'command.json',{'action':action})
                with self.assertRaises(ValueError):module.prepare_recipe(controller.path,root/'sources')
                self.assertEqual(controller.desired(),action)
                self.assertEqual(controller.snapshot()['active_seconds'],123)


if __name__=='__main__':unittest.main()
