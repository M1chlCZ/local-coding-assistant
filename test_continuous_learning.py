"""CPU checks for durable continuous control and historical accounting."""
import json
import tempfile
import unittest
from pathlib import Path

from continuous_learning import Controller, child_busy
from continuous_data import Ledger, source_manifest, verify_source
from train_adapter import atomic_json


class ContinuousTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parent = self.root/'parent'; self.parent.mkdir()
        self.status = {'status':'running', 'active_seconds':27000, 'phase':'collect', 'round':3,
                       'best_dev':{'tasks':[{'id':'a','passed':True}]}}
        atomic_json(self.parent/'status.json', self.status)
        atomic_json(self.parent/'command.json', {'action':'resume'})
        self.path = self.root/'continuous'
        self.controller = Controller.create(self.path, self.parent)

    def tearDown(self):
        self.temp.cleanup()

    def test_six_hour_setting_preserves_clocks_and_saved_work(self):
        self.status.update(status='paused',limit_seconds=43200,best_adapter='saved-checkpoint')
        atomic_json(self.parent/'status.json',self.status)
        self.controller.control('pause')
        self.controller.set_hours(6)
        reopened=Controller(self.path)
        self.assertEqual(reopened.state['experiment_limit_seconds'],21600)
        value=reopened.snapshot()
        self.assertEqual(value['child_limit_seconds'],21600)
        self.assertEqual(value['active_seconds'],27000)
        self.assertEqual(value['child_active_seconds'],27000)
        self.assertEqual(value['best_adapter'],'saved-checkpoint')
        self.assertEqual(value['best_dev'],self.status['best_dev'])
        self.assertEqual(value['desired'],'pause')
        self.assertEqual(value['child_status'],'paused')
        self.assertEqual(value['benchmark']['next_after_active_seconds'],0)
        self.assertIn('6 active training hours',value['benchmark']['schedule'])

    def test_duration_change_never_restarts_a_finished_or_stopped_session(self):
        for status in ('completed','stopped'):
            self.status.update(status=status,limit_seconds=43200)
            atomic_json(self.parent/'status.json',self.status)
            self.controller.control('stop')
            before=json.loads((self.parent/'status.json').read_text())
            self.controller.set_hours(6)
            self.assertEqual(json.loads((self.parent/'status.json').read_text()),before)
            self.assertEqual(self.controller.desired(),'stop')
            self.assertFalse(self.controller.should_launch())

    def test_duration_change_requires_an_idle_paused_worker(self):
        with self.assertRaises(ValueError):self.controller.set_hours(6)
        self.controller.control('pause')
        self.status['status']='paused';atomic_json(self.parent/'status.json',self.status)
        import fcntl
        with (self.parent/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(ValueError):self.controller.set_hours(6)
        for hours in (0,25,float('nan'),float('inf')):
            with self.assertRaises(ValueError):self.controller.set_hours(hours)

    def test_only_completed_experiments_can_validate_archived_trainer_sources(self):
        from unittest.mock import patch
        from training_data import sha256
        import learning_session as learning
        import continuous_learning as continuous
        runtime=self.root/'runtime';runtime.mkdir()
        (runtime/'learning_session.py').write_text('new trainer')
        (runtime/'training_feedback.py').write_text('new feedback')
        archive=self.path/'source-updates'/'reviewed'/'source-generation';archive.mkdir(parents=True)
        (archive/'learning_session.py').write_text('old trainer')
        old={'learning_session.py':sha256(archive/'learning_session.py')}
        self.status.update(status='completed',sources=old)
        atomic_json(self.parent/'status.json',self.status)
        self.controller.save(completed_source_generations={self.controller.state['child']:{'directory':str(archive),'sources':old}})
        atomic_json(self.path/'binding.json',{n:sha256(runtime/n) for n in ('learning_session.py','training_feedback.py')})
        with patch.object(continuous,'ROOT',runtime),patch.object(learning,'SOURCE_FILES',('learning_session.py','training_feedback.py')):
            self.controller.validate()
            self.status['status']='paused';atomic_json(self.parent/'status.json',self.status)
            with self.assertRaisesRegex(ValueError,'source changed'):self.controller.validate()
            self.status['status']='completed';atomic_json(self.parent/'status.json',self.status)
            (archive/'learning_session.py').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'source changed'):self.controller.validate()

    def test_pause_is_durable_and_forwarded_before_acknowledgement(self):
        self.controller.control('pause')
        self.assertEqual(json.loads((self.parent/'command.json').read_text())['action'], 'pause')
        self.assertEqual(self.controller.snapshot()['status'], 'pausing')
        reopened = Controller(self.path)
        self.assertEqual(reopened.desired(), 'pause')
        self.status['status']='paused';atomic_json(self.parent/'status.json',self.status)
        self.assertEqual(reopened.snapshot()['status'], 'paused')
        self.assertFalse(reopened.should_launch())

    def test_stop_survives_login_without_resuming_child(self):
        self.controller.control('stop')
        reopened = Controller(self.path)
        self.assertFalse(reopened.should_launch())
        self.assertEqual(json.loads((self.parent/'command.json').read_text())['action'], 'stop')
        self.assertEqual(reopened.desired(), 'stop')

    def test_repeated_status_reads_do_not_double_count_active_clock(self):
        self.assertEqual(self.controller.snapshot()['active_seconds'],27000)
        self.assertEqual(self.controller.snapshot()['active_seconds'],27000)
        self.status['active_seconds']=27008;atomic_json(self.parent/'status.json',self.status)
        self.assertEqual(Controller(self.path).snapshot()['active_seconds'],27008)
        next_child=self.root/'next';next_child.mkdir()
        atomic_json(next_child/'status.json',{'status':'running','active_seconds':10})
        self.controller.attach(next_child)
        self.assertEqual(self.controller.snapshot()['active_seconds'],27018)

    def test_completed_child_is_not_restarted(self):
        self.status['status']='completed';atomic_json(self.parent/'status.json',self.status)
        self.assertFalse(self.controller.should_launch())
        self.assertEqual(self.controller.snapshot()['status'],'running')

    def test_manual_child_pause_overrides_running_controller(self):
        atomic_json(self.parent/'command.json',{'action':'pause'})
        self.controller.sync_manual_control()
        self.assertEqual(self.controller.desired(),'pause')
        self.assertFalse(self.controller.should_launch())

    def test_retry_backoff_survives_restart_and_is_bounded(self):
        for attempt in range(1,7):
            self.controller.failure('Model server failed to start', now=100)
            self.controller=Controller(self.path)
            self.assertGreater(self.controller.state['retry_at'],100)
        self.assertEqual(self.controller.state['status'],'blocked')
        self.assertFalse(self.controller.should_launch(now=100000))
        self.controller.control('resume')
        self.assertEqual(self.controller.state['failures'],0)

    def test_integrity_failure_blocks_instead_of_retrying(self):
        self.controller.failure('Checkpoint files failed integrity verification',now=100)
        self.assertEqual(self.controller.state['status'],'blocked')
        self.assertFalse(self.controller.should_launch(now=100000))

    def test_worker_lock_detects_liveness_without_pid_reuse(self):
        import fcntl
        with (self.parent/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertTrue(child_busy(self.parent))
        self.assertFalse(child_busy(self.parent))

    def test_status_reader_cannot_overwrite_new_worker_state(self):
        stale=Controller(self.path)
        self.controller.save(detail='New worker decision',failures=3)
        stale.snapshot()
        self.assertEqual(Controller(self.path).state['failures'],3)

    def test_auxiliary_model_must_exit_before_pause_acknowledgement(self):
        self.status['status']='completed';atomic_json(self.parent/'status.json',self.status)
        self.controller.save(auxiliary_active=True)
        self.controller.control('pause')
        self.assertEqual(self.controller.snapshot()['status'],'pausing')
        self.controller.save(auxiliary_active=False)
        self.assertEqual(self.controller.snapshot()['status'],'paused')

    def test_retention_preserves_adopted_current_and_unfinished_sessions(self):
        children=self.path/'children';children.mkdir()
        for number in range(1,5):
            child=children/f'batch-{number:06d}';child.mkdir()
            atomic_json(child/'status.json',{'status':'completed' if number!=2 else 'paused'})
        self.controller.prune()
        self.assertTrue(self.parent.exists())
        self.assertFalse((children/'batch-000001').exists())
        self.assertTrue((children/'batch-000002').exists())
        self.assertTrue((children/'batch-000003').exists())
        self.assertTrue((children/'batch-000004').exists())

    def test_fresh_child_handoff_preserves_seed_and_global_clock(self):
        from unittest.mock import patch
        import learning_session as learning
        from train_adapter import CHECKPOINT_FILES, complete_checkpoint
        from training_data import sha256
        binding={'model':'Qwen/test','r':8}
        model=self.root/'adapter'/'checkpoint-5';model.mkdir(parents=True)
        for name in CHECKPOINT_FILES:(model/name).write_text(name)
        complete_checkpoint(model,binding);atomic_json(model.parent/'run.json',binding)
        anchor=self.parent/'round-001';anchor.mkdir()
        original={'id':'anchor-r1','repository':'anchor-r1','split':'train'}
        atomic_json(anchor/'tasks.json',[original])
        for name in ('teacher.json','training.jsonl','training.jsonl.manifest.json'):(anchor/name).write_text('{}')
        atomic_json(self.parent/'development-tasks.json',[{'id':'dev','repository':'dev','split':'dev'}])
        atomic_json(self.parent/'dev-base.json',{'tasks':[{'id':'dev','passed':False}]})
        self.status.update(status='completed',sources={n:sha256(learning.ROOT/n) for n in learning.SOURCE_FILES},
            best_adapter=str(model),research_adapter=str(model),research_score=1)
        atomic_json(self.parent/'status.json',self.status)
        tasks=[{'id':f'fresh-{i}','repository':f'fresh-{i}','split':'train' if i<32 else 'dev'} for i in range(52)]
        source=self.root/'fresh.json';atomic_json(source,tasks)
        # A completed parent keeps its original source generation; fresh children use the reviewed trainer.
        import shutil
        archive=self.path/'source-updates'/'completed-parent'/'source-generation';archive.mkdir(parents=True)
        old_sources=dict(self.status['sources']);old_sources.pop('training_feedback.py')
        for name in old_sources:
            target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(learning.ROOT/name,target)
        self.status['sources']=old_sources;atomic_json(self.parent/'status.json',self.status)
        self.controller.save(completed_source_generations={self.controller.state['child']:{'directory':str(archive),'sources':old_sources}})
        atomic_json(self.path/'binding.json',{n:sha256(learning.ROOT/n) for n in learning.SOURCE_FILES})
        child=self.controller.prepare_child(source)
        state=json.loads((child/'status.json').read_text())
        self.assertEqual(state['active_seconds'],0)
        self.assertEqual(state['limit_seconds'],21600)
        self.assertEqual(state['sources'],{n:sha256(learning.ROOT/n) for n in learning.SOURCE_FILES})
        self.assertEqual(self.controller.snapshot()['active_seconds'],27000)
        rows=json.loads((child/'round-003/tasks.json').read_text())
        self.assertEqual(len(rows),33)
        self.assertTrue(all(row['split']=='train' for row in rows))
        self.assertEqual(state['confirmation']['tasks'],20)
        self.assertEqual(sha256(Path(state['best_adapter'])/'adapter_model.safetensors'),sha256(model/'adapter_model.safetensors'))
        self.assertEqual(json.loads((self.parent/'status.json').read_text())['status'],'completed')

    def test_prefilled_curriculum_is_consumed_in_order_without_skipping_it(self):
        pool=self.controller.path/'pool'
        for number in (1,2):
            atomic_json(pool/f'batch-{number:06d}'/'tasks.json',[{'id':str(number)}])
            atomic_json(pool/f'batch-{number:06d}'/'manifest.json',{})
        self.assertEqual(self.controller.pending_tasks(),pool/'batch-000001/tasks.json')
        self.controller.save(last_consumed_sequence=1)
        self.assertEqual(self.controller.pending_tasks(),pool/'batch-000002/tasks.json')
        self.controller.save(last_consumed_sequence=2)
        self.assertIsNone(self.controller.pending_tasks())

    def test_confirmation_recovers_cached_tasks_and_reuses_identical_weights(self):
        from unittest.mock import patch
        import continuous_confirmation as confirmation
        from train_adapter import CHECKPOINT_FILES,complete_checkpoint
        from training_data import sha256
        model=self.root/'adapter'/'checkpoint-5';model.mkdir(parents=True)
        for name in CHECKPOINT_FILES:(model/name).write_text(name)
        complete_checkpoint(model,{});atomic_json(model.parent/'run.json',{})
        tasks=[{'id':str(n),'repository':str(n),'split':'dev','files':{'solution.py':'broken'},
            'editable':['solution.py'],'reference_patch':{'solution.py':'correct'}} for n in range(2)]
        atomic_json(self.parent/'confirmation-tasks.json',tasks)
        self.status.update(status='completed',best_adapter=str(model),confirmation={'consumed':False,
            'reserved_before_training':True,'tasks':2,'registry_sha256':sha256(self.parent/'confirmation-tasks.json'),
            'baseline_adapter':str(model),'baseline_adapter_sha256':sha256(model/'adapter_model.safetensors')})
        atomic_json(self.parent/'status.json',self.status)
        binding={'tasks_sha256':sha256(self.parent/'confirmation-tasks.json'),
            'baseline_sha256':sha256(model/'adapter_model.safetensors'),'current_sha256':sha256(model/'adapter_model.safetensors'),
            'settings':confirmation.SETTINGS,'limits':confirmation.LIMITS}
        output=self.controller.path/'confirmations'/self.parent.name
        for name in ('base','baseline'):
            atomic_json(output/f'{name}.json',{'binding':binding,'tasks':[{'id':str(n),'passed':n==0} for n in range(2)]})
        (self.root/'.cache/learning').mkdir(parents=True)
        with patch.object(self.controller,'validate'),patch.object(confirmation,'ROOT',self.root), \
             patch.object(confirmation,'grade',side_effect=lambda task,patch:{'passed':patch['solution.py']=='correct'}), \
             patch.object(confirmation.subprocess,'Popen',side_effect=AssertionError('Completed checks were repeated')), \
             patch.object(confirmation.socket,'socket') as socket:
            socket.return_value.__enter__.return_value.connect_ex.return_value=1
            confirmation.compare(self.controller)
        result=json.loads((output/'summary.json').read_text())
        self.assertTrue(result['completed']);self.assertTrue(result['same_weights_evaluated_once'])
        self.assertEqual(result['regressions'],[]);self.assertFalse(result['additional_gain_established'])


class SourceTests(unittest.TestCase):
    def test_source_manifest_is_frozen_and_corruption_rejected(self):
        value=source_manifest();self.assertEqual(len(value['shards']),50)
        self.assertEqual(value['revision'],'8f3ba5bafe4d6e8db46082cf7ae6741bc370604d')
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'shard';p.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'digest'):
                verify_source(p,value['shards'][0])

    def test_identity_ledger_excludes_old_rows_and_reserved_tasks_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'seen.sqlite';ledger=Ledger(path)
            t={'id':'x-r2','repository':'repo-r2','split':'dev','provenance':{'row_id':'old'}}
            self.assertTrue(ledger.reserve(t));ledger.close()
            ledger=Ledger(path)
            self.assertFalse(ledger.reserve({**t,'id':'different','repository':'different'}))
            self.assertFalse(ledger.reserve({**t,'id':'different','provenance':{'row_id':'new'}}))
            ledger.close()

    def test_prepared_batch_recovers_cursor_commit_without_regrading_or_reuse(self):
        from continuous_data import finish_batch, MANIFEST
        from training_data import sha256
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tasks=root/'tasks.json';manifest=root/'manifest.json'
            t={'id':'new','repository':'repo','split':'train','provenance':{'row_id':'new'}}
            atomic_json(tasks,[t]);atomic_json(manifest,{'source_manifest_sha256':sha256(MANIFEST),
                'tasks_sha256':sha256(tasks),'next_cursor':{'row':13000,'sequence':2}})
            ledger=Ledger(root/'seen.sqlite')
            finish_batch(manifest,root/'cursor.json',ledger)
            finish_batch(manifest,root/'cursor.json',ledger)
            self.assertEqual(json.loads((root/'cursor.json').read_text())['row'],13000)
            self.assertFalse(ledger.reserve(t));ledger.close()


if __name__=='__main__':unittest.main()
