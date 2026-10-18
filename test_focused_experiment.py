"""Finite research must survive interruption without repeating completed work."""
import fcntl
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


class FocusedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'session'

    def tearDown(self):
        self.tmp.cleanup()

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('focused_experiment'), 'Finite experiment controller missing')
        return importlib.import_module('focused_experiment')

    def create(self):
        module = self.module()
        config = {'schema': 1, 'inputs': {}, 'limit_seconds': 21600, 'stages': [
            {'name': 'baseline', 'command': [sys.executable, '-c', 'pass'], 'result': 'baseline.json'},
            {'name': 'train', 'command': [sys.executable, '-c', 'pass'], 'result': 'adapter/training.json'}]}
        return module.Experiment.create(self.path, config)

    def test_control_survives_reload_and_never_resets_clock(self):
        exp = self.create()
        exp.save(active_seconds=125, stage_index=1)
        exp.control('pause')
        other = self.module().Experiment(self.path)
        self.assertEqual(other.desired(), 'pause')
        self.assertEqual(other.state['active_seconds'], 125)
        other.control('resume')
        self.assertEqual(other.state['stage_index'], 1)

    def test_stale_running_state_is_not_reported_as_live(self):
        exp = self.create()
        exp.control('resume')
        exp.save(status='running', detail='Training')
        self.assertFalse(exp.snapshot()['worker_alive'])
        self.assertEqual(exp.snapshot()['status'], 'interrupted')
        with (self.path/'worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertTrue(exp.snapshot()['worker_alive'])
            self.assertEqual(exp.snapshot()['status'], 'running')

    def test_changed_bound_input_fails_before_gpu_work(self):
        exp = self.create()
        source = self.path/'input.json'; source.write_text('original')
        config = dict(exp.config, inputs={str(source): self.module().sha256(source)})
        other = self.module().Experiment.create(Path(self.tmp.name)/'bound', config)
        source.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            other.validate()

    def test_phase_advances_only_after_complete_result_and_checks_it_on_resume(self):
        exp = self.create()
        result = self.path/'baseline.json'
        result.write_text('{"completed": false}')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            exp.finish_stage()
        result.write_text('{"completed": true, "passed": 2}')
        exp.finish_stage()
        self.assertEqual(exp.state['stage_index'], 1)
        result.write_text('{"completed": true, "passed": 3}')
        with self.assertRaisesRegex(ValueError, 'Completed result changed'):
            exp.validate()

    def test_finished_or_stopped_experiment_never_restarts(self):
        exp = self.create()
        for state in ('completed', 'stopped', 'budget_exhausted'):
            exp.save(status=state)
            with self.assertRaisesRegex(ValueError, 'finished'):
                exp.control('resume')

    def test_stopped_continuous_session_requires_explicit_resume(self):
        from train_adapter import atomic_json
        exp = self.create()
        config = dict(exp.config, continuous=True, limit_seconds=None)
        atomic_json(self.path/'config.json', config)
        exp.save(config_sha256=self.module().sha256(self.path/'config.json'),
                 status='stopped', active_seconds=125, training_seconds=30)
        exp = self.module().Experiment(self.path)
        exp.control('stop')
        exp.run()  # Login recovery alone must not resume Stop.
        self.assertEqual(exp.snapshot()['status'], 'stopped')
        exp.control('resume')
        restored = self.module().Experiment(self.path).snapshot()
        self.assertEqual(restored['status'], 'interrupted')
        self.assertEqual(restored['desired'], 'resume')
        self.assertEqual(restored['active_seconds'], 125)
        self.assertEqual(restored['training_seconds'], 30)

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux worker restart test')
    def test_manual_resume_of_stopped_loop_keeps_completed_work(self):
        from train_adapter import atomic_json
        module = self.module()
        exp = self.create()
        (self.path/'baseline.json').write_text('{"completed": true}')
        exp.finish_stage()
        completed = list(exp.state['completed_stages'])
        result = self.path/'next.json'
        config = dict(exp.config, continuous=True, limit_seconds=None)
        config['stages'][1] = dict(name='next', gpu=False, result='next.json', command=[sys.executable,
            '-c', 'from pathlib import Path;Path('+repr(str(result))+').write_text(\'{"completed":true}\')'])
        atomic_json(self.path/'config.json', config)
        exp.save(config_sha256=module.sha256(self.path/'config.json'), status='stopped',
                 active_seconds=125, training_seconds=30)
        exp = module.Experiment(self.path)
        exp.control('resume')
        exp.run()
        self.assertEqual(exp.state['status'], 'completed')
        self.assertEqual(exp.state['completed_stages'][:1], completed)
        self.assertEqual(exp.state['training_seconds'], 30)
        self.assertGreaterEqual(exp.state['active_seconds'], 125)
        self.assertTrue(result.exists())

    def test_stage_result_cannot_escape_session(self):
        exp = self.create()
        bad = dict(exp.config, stages=[dict(exp.config['stages'][0], result='../result.json')])
        with self.assertRaisesRegex(ValueError, 'result path'):
            self.module().Experiment.create(Path(self.tmp.name)/'bad', bad)

    def test_failed_trainer_is_detected_only_for_current_process(self):
        from train_adapter import atomic_json
        exp = self.create()
        stage = dict(exp.config['stages'][1], train=True, progress='adapter/progress.json')
        progress = self.path/'adapter/progress.json'
        for pid, started, status, expected in [(71, 101, 'failed', 'CUDA failed'),
                (70, 101, 'failed', None), (71, 99, 'failed', None),
                (71, 101, 'running', None)]:
            atomic_json(progress, dict(pid=pid, started_at=started, status=status, error='CUDA failed'))
            self.assertEqual(exp.trainer_failure(stage, 71, 100), expected)
        self.assertIsNone(exp.trainer_failure(dict(stage, train=False), 71, 100))

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux stage process test')
    def test_invalid_stage_request_is_fatal_but_transient_failure_can_retry(self):
        exp=self.create();exp.control('resume')
        for code,error_type in [(2,ValueError),(1,RuntimeError)]:
            stage=dict(exp.config['stages'][0],gpu=False,
                       command=[sys.executable,'-c',f"import sys;print('request rejected');sys.exit({code})"])
            with self.assertRaisesRegex(error_type,'request rejected'):
                exp.execute_stage(stage)
            self.assertEqual(exp.state['stage_index'],0)

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux process ownership test')
    def test_failed_trainer_hung_in_cleanup_releases_its_process(self):
        module = self.module()
        exp = self.create()
        exp.control('resume')
        progress = self.path/'progress.json'
        script = ('import os,time,json;from pathlib import Path;p=Path('+repr(str(progress))+');'
            "tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps("
            "dict(status='failed',pid=os.getpid(),started_at=time.time(),error='CUDA failed')));tmp.replace(p);"
            'time.sleep(120)')
        stage = dict(exp.config['stages'][1], train=True, gpu=False,
                     command=[sys.executable, '-c', script])
        # A bounded child prevents a broken watchdog from hanging the test suite.
        driver = ('from focused_experiment import Experiment;'
                  'e=Experiment('+repr(str(self.path))+');e.execute_stage('+repr(stage)+')')
        result = subprocess.run([sys.executable, '-c', driver], capture_output=True,
                                text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('CUDA failed', result.stderr)
        self.assertFalse((self.path/'process.json').exists())
        self.assertEqual(module.Experiment(self.path).state['stage_index'], 0)

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux process ownership test')
    def test_finished_training_budget_allows_remaining_audit(self):
        module = self.module()
        result = self.path/'audit.json'
        config = {'inputs': {}, 'limit_seconds': 21600, 'stages': [
            {'name': 'audit', 'gpu': False, 'result': 'audit.json', 'command': [sys.executable, '-c',
             'from pathlib import Path;Path('+repr(str(result))+').write_text(\'{"completed": true}\')']} ]}
        exp = module.Experiment.create(self.path, config)
        exp.save(training_seconds=21600)
        exp.control('resume')
        process = subprocess.run([sys.executable, module.__file__, 'run', '--session', str(self.path)],
                                 capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr)
        final = module.Experiment(self.path)
        self.assertEqual(final.state['status'], 'completed')
        self.assertTrue(result.exists())
        self.assertEqual(final.state['training_seconds'], 21600)

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux process ownership test')
    def test_pause_resume_releases_child_and_keeps_completed_stage(self):
        module = self.module()
        exp = self.create()
        result = self.path/'baseline.json'
        result.write_text('{"completed": true}')
        exp.finish_stage()
        # A real interruptible subprocess tests Pause independently of model libraries.
        config = dict(exp.config)
        config['stages'][1] = {'name': 'collect', 'gpu': False,
            'command': [sys.executable, '-c',
                "import time;from pathlib import Path;p=Path("+repr(str(self.path))+ ");"
                "(p/'child-ready').touch();time.sleep(120)"], 'result': 'next.json'}
        from train_adapter import atomic_json
        atomic_json(self.path/'config.json', config)
        exp.save(config_sha256=module.sha256(self.path/'config.json'))
        exp.control('resume')
        process = subprocess.Popen([sys.executable, module.__file__, 'run', '--session', str(self.path)])
        try:
            deadline = time.monotonic()+15
            while not (self.path/'child-ready').exists() and time.monotonic() < deadline:
                time.sleep(.1)
            self.assertTrue((self.path/'child-ready').exists())
            module.Experiment(self.path).control('pause')
            while time.monotonic() < deadline:
                state = module.Experiment(self.path).snapshot()
                if state['status'] == 'paused':
                    break
                time.sleep(.1)
            self.assertEqual(state['status'], 'paused')
            self.assertEqual(state['stage_index'], 1)
            self.assertFalse((self.path/'process.json').exists())
            paused_clock = state['active_seconds']
            time.sleep(.3)
            self.assertEqual(module.Experiment(self.path).state['active_seconds'], paused_clock)
            (self.path/'child-ready').unlink()
            module.Experiment(self.path).control('resume')
            deadline = time.monotonic()+15
            while not (self.path/'child-ready').exists() and time.monotonic() < deadline:
                time.sleep(.1)
            self.assertTrue((self.path/'child-ready').exists(), 'Resume did not restart the unfinished stage')
            resumed = module.Experiment(self.path).snapshot()
            self.assertEqual(resumed['stage_index'], 1)
            self.assertGreaterEqual(resumed['active_seconds'], paused_clock)
            module.Experiment(self.path).control('stop')
            self.assertEqual(process.wait(timeout=10), 0)
        finally:
            if process.poll() is None:
                process.terminate(); process.wait(timeout=10)

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux process ownership test')
    def test_orphan_group_cleanup_works_after_leader_exits(self):
        module = self.module()
        exp = self.create()
        token = 'test-unique-process-token'
        script = "import subprocess,sys;subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])"
        leader = subprocess.Popen([sys.executable, '-c', script], start_new_session=True,
            env={**os.environ, 'LCA_FOCUSED_OWNER': str(self.path), 'LCA_FOCUSED_TOKEN': token})
        owner = {'pid': leader.pid, 'token': token}
        leader.wait(timeout=10)
        try:
            self.assertTrue(module.owned_group(owner, self.path))
            module.stop_owned_group(owner, self.path, seconds=2)
            self.assertFalse(module.owned_group(owner, self.path))
        finally:
            try: os.killpg(leader.pid, signal.SIGKILL)
            except ProcessLookupError: pass


if __name__ == '__main__':
    unittest.main()
