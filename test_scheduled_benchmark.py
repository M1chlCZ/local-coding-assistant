"""CPU checks for independent experiment-end audits and durable controls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from continuous_learning import Controller, ROOT, read
from research.student_benchmark import SOURCES, load_tasks, summarize
from train_adapter import atomic_json
from training_data import sha256


class EndLoop(BaseException):
    pass


class ScheduledBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.child = self.root/'experiment'
        self.adapter = self.root/'accepted'
        self.baseline = self.root/'baseline'
        for folder, content in ((self.adapter, 'current'), (self.baseline, 'starting')):
            folder.mkdir()
            (folder/'adapter_model.safetensors').write_text(content)
        atomic_json(self.child/'status.json', {'status': 'completed', 'round': 9,
            'active_seconds': 43200, 'limit_seconds': 43200, 'best_adapter': str(self.adapter),
            'confirmation': {'baseline_adapter': str(self.baseline)}})
        atomic_json(self.child/'command.json', {'action': 'resume'})
        self.controller = Controller.create(self.root/'continuous', self.child)
        atomic_json(self.controller.path/'confirmations'/self.child.name/'summary.json',
                    {'completed': True, 'regressions': []})

    def tearDown(self):
        self.temp.cleanup()

    def completed_report(self, output, adapter):
        _, tasks = load_tasks()
        binding = {'max_output_tokens': 1024, 'adapter_sha256': sha256(adapter/'adapter_model.safetensors'),
                   'sources': {name: sha256(ROOT/name) for name in SOURCES}}
        reports = {name: {'binding': binding, 'results': [
            {'task_id': task['task_id'], 'passed': i < count, 'elapsed_s': 1, 'output_tokens': 10}
            for i, task in enumerate(tasks)]} for name, count in (('base', 118), ('adapter', 121))}
        atomic_json(output/'binding.json', binding)
        for name, report in reports.items():
            atomic_json(output/f'{name}.json', report)
        atomic_json(output/'summary.json', summarize(binding, tasks, reports))
        atomic_json(output/'status.json', {'status': 'completed', 'completed': 164, 'total': 164})

    def test_full_audit_runs_after_confirmation_and_before_next_experiment(self):
        steps = []
        def process(command, log, forward=True):
            name = Path(command[1]).name
            steps.append(name)
            if name == 'student_benchmark.py':
                output = Path(command[command.index('--output')+1])
                self.completed_report(output, self.adapter)
            return True
        def prepare(_):
            steps.append('next-experiment')
            raise EndLoop()
        with patch.object(self.controller, 'validate'), patch.object(self.controller, 'recover_process'), \
             patch.object(self.controller, 'run_process', side_effect=process), \
             patch.object(self.controller, 'pending_tasks', return_value=self.root/'tasks.json'), \
             patch.object(self.controller, 'prepare_child', side_effect=prepare):
            with self.assertRaises(EndLoop):
                self.controller.run()
        self.assertEqual(steps, ['continuous_confirmation.py', 'student_benchmark.py', 'next-experiment'])
        self.assertEqual(self.controller.snapshot()['benchmark']['last']['adapter_passed'], 121)

    def test_confirmation_rollback_audits_the_adapter_retained_for_next_experiment(self):
        method = getattr(self.controller, 'benchmark_target', None)
        self.assertTrue(callable(method), 'Experiment-end benchmark selection is missing')
        atomic_json(self.controller.path/'confirmations'/self.child.name/'summary.json',
                    {'completed': True, 'regressions': ['lost-task']})
        adapter, output = method()
        self.assertEqual(adapter, self.baseline)
        self.assertEqual(output.name, sha256(self.baseline/'adapter_model.safetensors'))

    def test_audit_cannot_begin_before_training_and_confirmation_finish(self):
        method = getattr(self.controller, 'benchmark_target', None)
        self.assertTrue(callable(method), 'Experiment-end benchmark selection is missing')
        value = read(self.child/'status.json');value['status'] = 'running'
        atomic_json(self.child/'status.json', value)
        with self.assertRaises(ValueError):
            method()
        value['status'] = 'completed';atomic_json(self.child/'status.json', value)
        (self.controller.path/'confirmations'/self.child.name/'summary.json').unlink()
        with self.assertRaises(ValueError):
            method()

    def test_partial_or_tampered_report_cannot_become_last_complete_score(self):
        method = getattr(self.controller, 'record_benchmark', None)
        self.assertTrue(callable(method), 'Complete benchmark recording is missing')
        _, output = self.controller.benchmark_target()
        self.completed_report(output, self.adapter)
        report = read(output/'adapter.json');report['results'].pop()
        atomic_json(output/'adapter.json', report)
        with self.assertRaises(ValueError):
            method(output, reused=False)
        self.assertFalse((self.controller.path/'benchmarks/latest.json').exists())
        self.completed_report(output, self.adapter)
        summary = read(output/'summary.json');summary['models']['adapter']['passed'] = 164
        atomic_json(output/'summary.json', summary)
        with self.assertRaises(ValueError):
            method(output, reused=False)

    def test_same_weights_keep_same_resumable_folder_across_controller_restart(self):
        method = getattr(self.controller, 'benchmark_target', None)
        self.assertTrue(callable(method), 'Resumable benchmark target is missing')
        _, output = method()
        self.completed_report(output, self.adapter)
        self.controller.record_benchmark(output, reused=False)
        reopened = Controller(self.controller.path)
        self.assertEqual(reopened.benchmark_target()[1], output)
        reopened.record_benchmark(output, reused=True)
        last = reopened.snapshot()['benchmark']['last']
        self.assertTrue(last['reused']);self.assertEqual(last['base_passed'], 118)
        self.assertEqual(last['lost'], 0)
        self.assertEqual(last['adapter_sha256'], sha256(self.adapter/'adapter_model.safetensors'))

    def test_user_pause_interrupts_audit_without_starting_next_experiment(self):
        steps = []
        def process(command, log, forward=True):
            name = Path(command[1]).name;steps.append(name)
            if name == 'student_benchmark.py':
                self.assertFalse(forward)
                self.controller.control('pause')
                return False
            return True
        with patch.object(self.controller, 'validate'), patch.object(self.controller, 'recover_process'), \
             patch.object(self.controller, 'run_process', side_effect=process), \
             patch.object(self.controller, 'prepare_child', side_effect=AssertionError('Pause ignored')), \
             patch('continuous_learning.time.sleep', side_effect=EndLoop):
            with self.assertRaises(EndLoop):
                self.controller.run()
        self.assertEqual(steps, ['continuous_confirmation.py', 'student_benchmark.py'])
        self.assertEqual(Controller(self.controller.path).desired(), 'pause')
        self.assertFalse((self.controller.path/'benchmarks/latest.json').exists())
        self.assertEqual(self.controller.snapshot()['active_seconds'], 43200)


if __name__ == '__main__':
    unittest.main()
