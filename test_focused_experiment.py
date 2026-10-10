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

    def test_stage_result_cannot_escape_session(self):
        exp = self.create()
        bad = dict(exp.config, stages=[dict(exp.config['stages'][0], result='../result.json')])
        with self.assertRaisesRegex(ValueError, 'result path'):
            self.module().Experiment.create(Path(self.tmp.name)/'bad', bad)

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
