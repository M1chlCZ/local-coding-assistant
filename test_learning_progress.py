"""Keep worker state distinct from finished experiment and benchmark progress."""
import copy
import importlib
import importlib.util
import unittest


class LearningProgressTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
