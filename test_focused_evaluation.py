import importlib
import importlib.util
import unittest


class FocusedEvaluationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('focused_evaluation'))
        return importlib.import_module('focused_evaluation')

    def test_partial_or_reordered_results_cannot_become_complete(self):
        module = self.module()
        tasks = {'go': [{'id': 'a'}, {'id': 'b'}]}
        for rows in ([{'id': 'a', 'passed': True}],
                     [{'id': 'b', 'passed': True}, {'id': 'a', 'passed': True}]):
            with self.assertRaises(ValueError):
                module.summarize({}, tasks, {'go': rows})

    def test_summary_counts_functional_passes_and_equal_language_weight(self):
        module = self.module()
        tasks = {'go': [{'id': 'a'}], 'typescript': [{'id': 'b'}, {'id': 'c'}]}
        report = {'go': [{'id': 'a', 'passed': True}],
                  'typescript': [{'id': 'b', 'passed': False}, {'id': 'c', 'passed': True}]}
        result = module.summarize({'engine': 'qwen35'}, tasks, report)
        self.assertEqual((result['passed'], result['total']), (2, 3))
        self.assertEqual(result['macro_pass_at_1'], .75)
        self.assertTrue(result['completed'])

    def test_report_cannot_count_transport_failure_as_model_failure(self):
        module = self.module()
        with self.assertRaisesRegex(ValueError, 'infrastructure'):
            module.summarize({}, {'go': [{'id': 'a'}]}, {'go': [{'id': 'a', 'error': '503'}]})


if __name__ == '__main__':
    unittest.main()
