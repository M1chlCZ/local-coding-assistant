"""Keep worker state distinct from finished experiment and benchmark progress."""
import copy
import fcntl
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class LearningProgressTests(unittest.TestCase):
    def test_unlocked_worker_does_not_display_stale_running_status(self):
        module = importlib.import_module('learning_progress')
        self.assertTrue(hasattr(module, 'worker_health'), 'Worker liveness is not checked')
        value = {**self.value, 'desired': 'resume'}
        result = module.worker_health(value, False)
        self.assertEqual(result['status'], 'interrupted')
        self.assertEqual(result['saved_status'], 'running')
        self.assertEqual(result['desired'], 'resume')
        self.assertFalse(result['worker_alive'])
        self.assertIn('Resume', result['detail'])
        self.assertEqual(value['status'], 'running')

    def test_pause_and_stop_remain_saved_when_worker_is_absent(self):
        module = importlib.import_module('learning_progress')
        self.assertTrue(hasattr(module, 'worker_health'), 'Worker liveness is not checked')
        for action in ('pause', 'stop'):
            result = module.worker_health({**self.value, 'desired': action}, False)
            self.assertEqual(result['desired'], action)
            self.assertIn(action, result['detail'].lower())
            self.assertNotIn('Click Resume', result['detail'])

    def test_live_and_finished_states_are_preserved(self):
        module = importlib.import_module('learning_progress')
        self.assertTrue(hasattr(module, 'worker_health'), 'Worker liveness is not checked')
        self.assertEqual(module.worker_health(self.value, True)['status'], 'running')
        for status in ('completed', 'paused', 'stopped', 'blocked', 'failed'):
            self.assertEqual(module.worker_health({**self.value, 'status': status}, False)['status'], status)

    def test_liveness_checks_real_worker_lock_without_creating_files(self):
        module = importlib.import_module('learning_progress')
        self.assertTrue(hasattr(module, 'worker_alive'), 'Worker lock inspection is missing')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertFalse(module.worker_alive(path))
            self.assertEqual(list(path.iterdir()), [])
            with (path/'worker.lock').open('w') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(module.worker_alive(path))
            self.assertFalse(module.worker_alive(path))

    def view(self, value, state, confirmation=None):
        self.assertIsNotNone(importlib.util.find_spec('learning_progress'),
                             'Read-only continuous progress view is missing')
        return importlib.import_module('learning_progress').progress_view(value, state, confirmation)

    def setUp(self):
        self.value = {'status': 'running', 'child_status': 'completed', 'phase': 'collect',
            'detail': 'Active time limit reached; progress retained', 'active_seconds': 43202,
            'confirmation': {'tasks': 20}, 'benchmark': {'progress': None}}
        self.state = {'auxiliary_active': True, 'detail': 'Completing reserved matched confirmation checks'}

    def test_completed_experiment_displays_live_reserved_checks(self):
        result = self.view(self.value, self.state,
            {'status': 'running', 'mode': 'base', 'completed': 12, 'total': 20, 'task': 'fresh-task'})
        self.assertEqual(result['status'], 'running')
        self.assertEqual(result['phase'], 'confirmation')
        self.assertEqual(result['confirmation_progress']['completed'], 12)
        self.assertNotIn('time limit', result['detail'])
        self.assertEqual(result['active_seconds'], 43202)

    def test_reference_audit_is_visible_before_first_model_task(self):
        result = self.view(self.value, self.state)
        self.assertEqual(result['confirmation_progress']['phase'], 'reference audit')
        self.assertEqual(result['confirmation_progress']['total'], 20)

    def test_read_only_view_does_not_modify_worker_snapshot(self):
        original = copy.deepcopy(self.value)
        self.view(self.value, self.state, {'mode': 'baseline', 'completed': 1, 'total': 20})
        self.assertEqual(self.value, original)

    def test_pause_and_stop_are_preserved_during_handoff(self):
        for status in ('pausing', 'paused', 'stopping', 'stopped'):
            value = {**self.value, 'status': status}
            result = self.view(value, self.state, {'mode': 'base', 'completed': 12, 'total': 20})
            self.assertEqual(result['status'], status)

    def test_full_benchmark_displays_its_own_phase(self):
        value = {**self.value, 'benchmark': {'progress': {'status': 'running', 'phase': 'adapter',
            'completed': 42, 'total': 164}}}
        result = self.view(value, {**self.state, 'detail': 'Scheduled HumanEval comparison'})
        self.assertEqual(result['phase'], 'benchmark')
        self.assertNotIn('time limit', result['detail'])

    def test_real_failure_remains_visible_with_its_actual_reason(self):
        value = {**self.value, 'status': 'blocked'}
        reason = 'Confirmation server failed to start'
        result = self.view(value, {'auxiliary_active': False, 'detail': reason})
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['detail'], reason)

    def test_saved_benchmark_activity_is_read_only_and_counts_both_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'binding.json').write_text(json.dumps({'languages': ['python', 'go']}))
            for name, rows in {'base-python': [{'id': 'p1'}, {'id': 'p2'}],
                               'base-go': [{'id': 'g1'}, {'id': 'g2'}],
                               'adapter-python': [{'id': 'p1'}, {'id': 'p2'}],
                               'adapter-go': [{'id': 'g1', 'passed': False}]}.items():
                (path/(name+'.json')).write_text(json.dumps(rows))
            before = {p.name: p.read_bytes() for p in path.iterdir()}
            module = importlib.import_module('learning_progress')
            self.assertTrue(hasattr(module, 'benchmark_activity'), 'Saved benchmark activity is missing')
            result = module.benchmark_activity(path, {'phase': 'adapter', 'language': 'go'})
            self.assertEqual(result['stage'], 4)
            self.assertEqual(result['stages'], 4)
            self.assertEqual(result['overall_completed'], 7)
            self.assertEqual(result['overall_total'], 8)
            self.assertEqual(result['last_result'], {'task': 'g1', 'passed': False})
            self.assertEqual({p.name: p.read_bytes() for p in path.iterdir()}, before)

    def test_incomplete_base_reports_do_not_invent_an_overall_total(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'binding.json').write_text(json.dumps({'languages': ['python', 'go']}))
            (path/'base-python.json').write_text(json.dumps([{'id': 'p1', 'passed': True}]))
            module = importlib.import_module('learning_progress')
            self.assertTrue(hasattr(module, 'benchmark_activity'), 'Saved benchmark activity is missing')
            result = module.benchmark_activity(path, {'phase': 'base', 'language': 'python'})
            self.assertEqual(result['stage'], 1)
            self.assertEqual(result['overall_completed'], 1)
            self.assertNotIn('overall_total', result)

    def test_old_single_language_benchmark_still_displays_its_own_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            module = importlib.import_module('learning_progress')
            self.assertTrue(hasattr(module, 'benchmark_activity'), 'Saved benchmark activity is missing')
            self.assertEqual(module.benchmark_activity(Path(directory), {'phase': 'base'}), {})


if __name__ == '__main__':
    unittest.main()
