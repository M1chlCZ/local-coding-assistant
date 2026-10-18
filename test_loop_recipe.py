"""Regression checks for useful training blocks and bounded evaluation overhead."""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from coding_loop import CodingLoop
import coding_loop_stage as stage
from qwen35_train import atomic_json, sha256


class RecipeTests(unittest.TestCase):
    def test_export_keeps_verified_majority_language_examples(self):
        with tempfile.TemporaryDirectory() as d:
            session=Path(d); folder=session/'round-000001'; folder.mkdir()
            tasks=[]; answers=[]
            for lang,count in zip(stage.LANGUAGES,[28,7,15,6,11]):
                for i in range(count):
                    key=f'{lang}-{i}'
                    tasks.append(dict(id=key,language=lang,repository=key,split='train',cases=[],
                        provenance=dict(dataset='test',revision='a'*40)))
                    answers.append(dict(id=key,passed=True,answer='valid source'))
            atomic_json(folder/'tasks.json',dict(train=tasks,validation=[]))
            atomic_json(folder/'prepare.json',dict(tasks_sha256=sha256(folder/'tasks.json')))
            atomic_json(folder/'collect.json',dict(rows=answers))
            tokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:object())
            with patch.dict(sys.modules,transformers=types.SimpleNamespace(AutoTokenizer=tokenizer)), \
                 patch('qwen35_train.encode_rows',return_value=([{}],{})), \
                 patch.object(stage,'messages',return_value=[dict(role='user',content='task')]), \
                 patch.object(stage,'check_program',return_value=dict(passed=True)), \
                 patch.object(stage,'replay_runtime',return_value={'image':'fixture'}), \
                 patch.object(stage,'progress'):
                result=stage.export_data(session,folder,dict(replay_per_language=128,min_examples=60,min_per_language=5))
            self.assertEqual(result['rows'],67,'Valid examples must not be cut to the smallest language')
            self.assertTrue(result['ready'])

    def test_new_sessions_require_substantial_training_data(self):
        with tempfile.TemporaryDirectory() as d, patch('coding_loop.SOURCES',()):
            loop=CodingLoop.prepare(Path(d)/'loop')
            self.assertGreaterEqual(loop.config.get('min_examples',20),100)
            self.assertEqual(loop.config.get('max_epochs',1),1)
            self.assertGreaterEqual(loop.config['steps_per_round'],64)

    def test_nonpromising_fifth_cycle_does_not_force_full_audit(self):
        # The periodic counter tracks actual updates, not collection-only cycles.
        with tempfile.TemporaryDirectory() as d:
            session=Path(d); folder=session/'round-000005';folder.mkdir()
            for n in range(1,6):
                atomic_json(session/f'round-{n:06d}/adapter/training.json',
                    dict(status='completed',skipped=n<4))
            self.assertFalse(stage.audit_due(session,folder,dict(promising=False),dict(audit_every_updates=20)))
            self.assertTrue(stage.audit_due(session,folder,dict(promising=True),dict(audit_every_updates=20)))
            self.assertFalse(stage.audit_due(session,folder,dict(skipped=True),dict(audit_every_updates=20)))
            self.assertTrue(stage.audit_due(session,folder,dict(promising=False),dict(audit_every_updates=2)))

    def test_replay_check_cache_is_bound_to_answer_tests_and_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            session=Path(d);task=dict(language='go',cases=[dict(input='1',output='2')])
            with patch.object(stage,'replay_runtime',return_value={'image':'a'}) as runtime, \
                 patch.object(stage,'check_program',return_value=dict(passed=True)) as check:
                self.assertTrue(stage.verify_replay(session,task,'code'))
                self.assertTrue(stage.verify_replay(session,task,'code'))
                self.assertEqual(check.call_count,1)
                stage.verify_replay(session,task,'other code')
                task['cases'][0]['output']='3';stage.verify_replay(session,task,'code')
                runtime.return_value={'image':'b'};stage.verify_replay(session,task,'code')
                self.assertEqual(check.call_count,4)

    def test_started_audit_remains_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            session=Path(d);folder=session/'round-000005'
            atomic_json(folder/'audit-parent/binding.json',dict(mode='base'))
            self.assertTrue(stage.audit_due(session,folder,dict(promising=False),dict(audit_every_updates=20)))

    def test_seed_tamper_or_validation_family_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            session=Path(d);folder=session/'round-000001';folder.mkdir()
            seed=session/'seed.json';atomic_json(seed,[])
            cfg=dict(replay_seed=dict(file='seed.json',sha256='0'*64))
            tokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:object())
            with patch.dict(sys.modules,transformers=types.SimpleNamespace(AutoTokenizer=tokenizer)):
                with self.assertRaisesRegex(ValueError,'seed changed'):
                    stage.export_data(session,folder,cfg)
                atomic_json(seed,[dict(task=dict(split='dev',provenance=dict(row_id='a'*64)),answer={})])
                cfg['replay_seed']['sha256']=sha256(seed)
                with self.assertRaisesRegex(ValueError,'training families only'):
                    stage.export_data(session,folder,cfg)

    def test_train_block_uses_saved_recipe_and_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);data=folder/'train.jsonl';data.write_text('data')
            atomic_json(folder/'export.json',dict(ready=True,data_sha256=sha256(data)))
            cfg=dict(steps_per_round=128,max_epochs=2,learning_rate=2e-5,min_examples=100,min_per_language=8)
            args=types.SimpleNamespace(pause_file=folder/'pause.requested',resume=True)
            with patch('qwen35_train.main') as main,patch.object(sys,'argv',[]):
                stage.train_block(folder,folder,args,cfg,None)
                command=list(sys.argv)
            self.assertEqual(command[command.index('--max-epochs')+1],'2')
            self.assertEqual(command[command.index('--min-examples')+1],'100')
            self.assertIn('--resume',command)
            self.assertEqual(command[command.index('--pause-file')+1],str(args.pause_file))

    def test_full_regression_reduces_next_update_with_a_floor(self):
        with tempfile.TemporaryDirectory() as d:
            session=Path(d);old=session/'round-000010';folder=session/'round-000011'
            cfg=dict(steps_per_round=128,max_epochs=1,learning_rate=2e-5)
            run=old/'adapter/run.json'
            def evidence(rate,lost):
                atomic_json(run,dict(learning_rate=rate))
                atomic_json(old/'finish.json',dict(completed=True,accepted=not lost,
                    comparisons=dict(training_effect=dict(audit=dict(go=dict(lost=lost)))),
                    inputs={str(run.relative_to(session)):sha256(run)}))
            evidence(2e-5,['lost-task'])
            recipe=stage.training_recipe(session,folder,cfg)
            self.assertEqual(recipe['learning_rate'],1e-5)
            self.assertEqual(recipe['max_epochs'],1)
            evidence(1e-5,['lost-task'])
            self.assertEqual(stage.training_recipe(session,folder,cfg)['learning_rate'],5e-6)
            evidence(5e-6,['lost-task'])
            self.assertEqual(stage.training_recipe(session,folder,cfg)['learning_rate'],5e-6)
            evidence(2e-6,['lost-task'])
            self.assertEqual(stage.training_recipe(session,folder,cfg)['learning_rate'],2e-6)
            evidence(1e-5,[])
            self.assertEqual(stage.training_recipe(session,folder,cfg)['learning_rate'],1e-5)
            run.write_text('{}')
            with self.assertRaisesRegex(ValueError,'recipe evidence changed'):
                stage.training_recipe(session,folder,cfg)

    def test_backoff_ignores_incomplete_and_validation_only_reports(self):
        with tempfile.TemporaryDirectory() as d:
            session=Path(d);folder=session/'round-000003'
            cfg=dict(steps_per_round=128,max_epochs=1,learning_rate=2e-5)
            atomic_json(session/'round-000001/finish.json',dict(completed=False,
                comparisons=dict(training_effect=dict(audit=dict(go=dict(lost=['x']))))))
            atomic_json(session/'round-000002/finish.json',dict(completed=True,
                comparisons=dict(training_effect=dict(validation=dict(go=dict(lost=['x']))))))
            self.assertEqual(stage.training_recipe(session,folder,cfg)['learning_rate'],2e-5)

    def test_pause_resume_keeps_the_started_training_recipe(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'round-000011'
            saved=dict(max_steps=128,max_epochs=2,learning_rate=2e-5)
            atomic_json(folder/'adapter/request.json',saved)
            recipe=stage.training_recipe(Path(d),folder,dict(steps_per_round=64,max_epochs=1,learning_rate=5e-6))
            self.assertEqual({k:recipe[k] for k in saved},saved)

if __name__=='__main__':unittest.main()
