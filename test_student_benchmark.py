"""Protect matched benchmark completeness and resumable comparisons without a GPU."""
import json
import tempfile
import unittest
from pathlib import Path

from research.student_benchmark import load_report, load_tasks, summarize
from research import student_benchmark


class StudentBenchmarkTests(unittest.TestCase):
    def test_only_identified_paused_worker_can_be_released(self):
        guard = getattr(student_benchmark, 'paused_worker_pid', None)
        self.assertTrue(callable(guard), 'Paused-worker ownership handoff is missing')
        owner = {'pid': 123, 'start': '456', 'forward': True}
        command = ['python', 'learning_session.py', 'run']
        environment = [b'LCA_CONTINUOUS_OWNER=/controller']
        self.assertEqual(guard('pause', 'paused', owner, '456', 123, environment, command, '/controller'), 123)
        for action, status, start, group, env in [
                ('resume', 'paused', '456', 123, environment),
                ('pause', 'running', '456', 123, environment),
                ('pause', 'paused', 'new-pid', 123, environment),
                ('pause', 'paused', '456', 999, environment),
                ('pause', 'paused', '456', 123, [b'LCA_CONTINUOUS_OWNER=/other-project'])]:
            with self.assertRaises(ValueError):
                guard(action, status, owner, start, group, env, command, '/controller')

    def test_complete_pinned_registry(self):
        metadata, tasks = load_tasks()
        self.assertEqual(len(tasks), 164)
        self.assertEqual([t['task_id'] for t in tasks], [f'HumanEval/{i}' for i in range(164)])
        self.assertEqual(metadata['repo'], 'openai/human-eval')

    def test_resume_rejects_different_weights_or_task_order(self):
        tasks = [{'task_id': 'a'}, {'task_id': 'b'}]
        binding = {'adapter_sha256': 'first'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'base.json'
            path.write_text(json.dumps({'binding': binding, 'results': [{'task_id': 'a'}]}))
            self.assertEqual(len(load_report(path, binding, tasks)['results']), 1)
            with self.assertRaises(ValueError):
                load_report(path, {'adapter_sha256': 'second'}, tasks)
            with self.assertRaises(ValueError):
                load_report(path, binding, list(reversed(tasks)))

    def test_partial_result_cannot_be_a_score(self):
        tasks = [{'task_id': 'a'}, {'task_id': 'b'}]
        reports = {'base': {'results': []}, 'adapter': {'results': [{'task_id': 'a'}]}}
        with self.assertRaises(ValueError):
            summarize({'max_output_tokens': 1024}, tasks, reports)

    def test_higher_score_still_records_lost_tasks(self):
        tasks = [{'task_id': name} for name in ('a', 'b', 'c')]
        def rows(flags):
            return {'results': [{'task_id': t['task_id'], 'passed': flag, 'elapsed_s': 2, 'output_tokens': 10}
                                for t, flag in zip(tasks, flags)]}
        summary = summarize({'max_output_tokens': 1024}, tasks,
                            {'base': rows([True, False, False]), 'adapter': rows([False, True, True])})
        self.assertTrue(summary['adapter_vs_base']['higher_score'])
        self.assertFalse(summary['adapter_vs_base']['no_regression'])
        self.assertEqual(summary['adapter_vs_base']['lost'], ['a'])
        self.assertEqual(summary['adapter_vs_base']['gained'], ['b', 'c'])


if __name__ == '__main__':
    unittest.main()
