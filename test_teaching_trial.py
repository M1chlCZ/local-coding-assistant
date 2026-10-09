import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import learning_session as learning
from train_adapter import atomic_json,parser
from training_data import sha256


class TeachingTrialTests(unittest.TestCase):
    def test_both_arms_finish_with_complete_checks_before_confirmation(self):
        from train_adapter import CHECKPOINT_FILES,complete_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            session=learning.Session(Path(directory)/'session',6,Path('/teacher.gguf'),research=True)
            self.addCleanup(session.lock.close);self.addCleanup(session.gpu_lock.close)
            tasks=[{'id':str(i)} for i in range(3)]
            base={'tasks':[{'id':t['id'],'passed':False} for t in tasks]}
            origin={'tasks':[{'id':t['id'],'passed':i==0} for i,t in enumerate(tasks)]}
            session.state.update(recipe='balanced-code-v1',recipe_trials=True,teaching_trial=True,fast_reject=True,
                round=1,best_adapter='/origin',best_dev=origin,trial_origin_adapter='/origin',trial_origin_dev=origin)
            folder=session.path/'round-001';folder.mkdir();atomic_json(session.path/'dev-base.json',base)
            for index in range(2):
                output=session.training_output(folder,index);checkpoint=output/'checkpoint-1';checkpoint.mkdir(parents=True)
                binding={'dataset_sha256':str(index)*64,'warm_start_sha256':'b'*64};atomic_json(output/'run.json',binding)
                for name in CHECKPOINT_FILES:(checkpoint/name).write_text('fixture')
                complete_checkpoint(checkpoint,binding)
                rows=[{'id':t['id'],'passed':i in ({0,1} if index==0 else {0,2})} for i,t in enumerate(tasks)]
                with patch.object(session,'server',return_value=True),patch.object(session,'stop_child'), \
                     patch.object(session,'interrupted',return_value=False),patch.object(learning,'development_tasks',return_value=tasks), \
                     patch.object(learning,'quality_rows',side_effect=lambda state,chunk,url:[r for r in rows if r['id'] in {t['id'] for t in chunk}]),patch('evaluation.request'):
                    session.evaluate(folder)
                result=json.loads((folder/('result-'+output.name+'.json')).read_text())
                self.assertTrue(result['complete']);self.assertEqual(result['evaluated'],3)
                self.assertEqual(result['accepted'],index==0)
                self.assertEqual(session.state['trial_origin_adapter'],'/origin')
                self.assertEqual(session.state['round'],1)
                self.assertEqual(session.state['status'],'running' if index==0 else 'completed')
            self.assertEqual(len(session.state['completed_rounds']),2)
            self.assertEqual(session.state['completion_reason'],'teaching_trial_completed')

    def test_token_filter_drops_overlength_pair_from_both_arms(self):
        import sys
        from types import SimpleNamespace
        from teaching_trial import filter_pairs
        from code_recipe import export_code,LANGUAGES
        from test_code_recipe import CodeRecipeTests
        from training_data import load_verified
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);tasks=[CodeRecipeTests().task(l,i) for l in LANGUAGES for i in range(2)]
            path=folder/'tasks.json';atomic_json(path,tasks)
            for arm in ('control','correction'):
                report={'schema_version':1,'split':'train','tasks_sha256':sha256(path),'tasks':[
                    {**{k:t[k] for k in ('id','repository','split')},'mode':'direct','passed':True,
                     'patch':{t['editable'][0]:'LONG' if arm=='correction' and t==tasks[0] else 'valid'}} for t in tasks]}
                report_path=folder/(arm+'.json');atomic_json(report_path,report)
                with patch('code_recipe.grade',return_value={'passed':True}):
                    export_code([report_path],path,folder/f'raw-{arm}.jsonl')
            tokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:None)
            with patch.dict(sys.modules,{'transformers':SimpleNamespace(AutoTokenizer=tokenizer)}), \
                 patch('teaching_trial.encode_row',side_effect=lambda t,r,n: None if r['messages'][-1]['content']=='LONG' else {'labels':[1]}):
                filter_pairs(folder)
            fit=json.loads((folder/'paired-fit.json').read_text())
            self.assertEqual(len(fit['ids']),9);self.assertNotIn(tasks[0]['id'],fit['ids'])
            self.assertEqual(len(load_verified(folder/'raw-control.jsonl',path)),10)

    def test_dataset_recovers_partial_writes_and_skips_identical_filtered_targets(self):
        from types import SimpleNamespace
        from teaching_trial import dataset,filter_pairs
        from code_recipe import LANGUAGES
        from test_code_recipe import CodeRecipeTests
        from training_data import manifest_path,load_verified
        from unittest.mock import Mock
        import sys
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);tasks=[CodeRecipeTests().task(l,i) for l in LANGUAGES for i in range(2)]
            path=folder/'tasks.json';atomic_json(path,tasks)
            rows=[{**{k:t[k] for k in ('id','repository','split')},'mode':'direct','passed':True,
                   'patch':{t['editable'][0]:'valid'}} for t in tasks]
            atomic_json(folder/'teacher.json',{'schema_version':1,'split':'train','tasks_sha256':sha256(path),'tasks':rows})
            atomic_json(folder/'student.json',{'tasks':[{**rows[0],'passed':False}]})
            atomic_json(folder/'corrections.json',{'tasks':[{**rows[0],'patch':{tasks[0]['editable'][0]:'fixed'}}]})
            (folder/'raw-control.jsonl').write_text('partial')
            manifest_path(folder/'raw-correction.jsonl').write_text('partial')
            (folder/'training-control.jsonl').write_text('partial')
            manifest_path(folder/'training-correction.jsonl').write_text('partial')
            session=Mock();session.interrupted.return_value=False
            session.child.poll.return_value=0;session.child.returncode=0
            session.spawn.side_effect=lambda *args:filter_pairs(folder)
            tokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:None)
            with patch.dict(sys.modules,{'transformers':SimpleNamespace(AutoTokenizer=tokenizer)}), \
                 patch('teaching_trial.encode_row',return_value={'labels':[1]}),patch('code_recipe.grade',return_value={'passed':True}):
                dataset(session,folder)
                session.save.assert_called_with(phase='train',trial_index=0,teaching_pairs=1,teaching_examples=10,
                    detail='Matched teaching trial: 10 examples, 1 corrected targets')
                self.assertEqual(len(list(folder.glob('*.incomplete-*'))),4)
                output=folder/'training-control.jsonl';modified=output.stat().st_mtime_ns
                self.assertEqual(len(load_verified(output,path)),10)
                dataset(session,folder);self.assertEqual(output.stat().st_mtime_ns,modified)
                fit=json.loads((folder/'paired-fit.json').read_text());fit['ids'].remove(tasks[0]['id'])
                atomic_json(folder/'paired-fit.json',fit);dataset(session,folder)
                self.assertEqual(session.save.call_args.kwargs['completion_reason'],'no_verified_corrections')
                fit['ids'].insert(0,tasks[0]['id']);atomic_json(folder/'paired-fit.json',fit)
                output.write_text('corrupted')
                with self.assertRaises(ValueError):dataset(session,folder)

    def test_trials_change_targets_only_and_keep_the_initial_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            session=learning.Session(Path(directory)/'session',6,Path('/teacher.gguf'),research=True)
            self.addCleanup(session.lock.close);self.addCleanup(session.gpu_lock.close)
            session.state.update(recipe='balanced-code-v1',recipe_trials=True,teaching_trial=True,round=1,
                best_adapter='/later-winner',trial_origin_adapter='/fixed-origin',trial_index=0)
            first=parser().parse_args(session.training_command(session.path/'round-001')[2:])
            session.state['trial_index']=1
            second=parser().parse_args(session.training_command(session.path/'round-001')[2:])
            self.assertEqual(first.warm_start,Path('/fixed-origin'));self.assertEqual(first.warm_start,second.warm_start)
            self.assertEqual(first.max_steps,second.max_steps);self.assertEqual(first.learning_rate,second.learning_rate)
            self.assertEqual(first.max_epochs,second.max_epochs);self.assertEqual(first.seed,second.seed)
            self.assertNotEqual(first.dataset,second.dataset)
            self.assertEqual(len(session.trial_names()),2)

    def test_only_verified_corrections_of_student_failures_replace_targets(self):
        from teaching_trial import paired_reports
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tasks.json'
            tasks=[{'id':str(i),'repository':str(i),'split':'train'} for i in range(3)]
            atomic_json(path,tasks)
            rows=[{**t,'mode':'direct','passed':True,'patch':{'source':'teacher-'+t['id']}} for t in tasks]
            teacher={'schema_version':1,'split':'train','tasks_sha256':sha256(path),'tasks':rows}
            student=[{**r,'passed':i==0} for i,r in enumerate(rows)]
            corrected=[{**r,'passed':i!=2,'patch':{'source':'repair-'+r['id']}} for i,r in enumerate(rows)]
            result,count=paired_reports(path,teacher,student,corrected)
            self.assertEqual(count,1)
            self.assertEqual([r['patch']['source'] for r in result['tasks']],['teacher-0','repair-1','teacher-2'])
            self.assertEqual(teacher['tasks'][1]['patch']['source'],'teacher-1')
            with self.assertRaises(ValueError):paired_reports(path,teacher,student,corrected+[{'id':'unknown'}])
            tasks[0]['split']='dev';atomic_json(path,tasks)
            teacher['tasks_sha256']=sha256(path)
            with self.assertRaises(ValueError):paired_reports(path,teacher,student,corrected)

    def test_collection_resume_binds_saved_student_to_exact_weights(self):
        from teaching_trial import student_report
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tasks=root/'tasks.json';atomic_json(tasks,[{'id':'a','repository':'a','split':'train'}])
            weights=root/'weights';weights.write_text('old');report=root/'student.json'
            value=student_report(report,tasks,weights,['a'])
            value['tasks']=[{'id':'a'}];atomic_json(report,value)
            self.assertEqual(student_report(report,tasks,weights,['a'])['tasks'],[{'id':'a'}])
            weights.write_text('new')
            with self.assertRaises(ValueError):student_report(report,tasks,weights,['a'])


if __name__=='__main__':unittest.main()
