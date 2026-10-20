import importlib
import unittest

from test_structured_output import response


class ComparisonTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.structured_compare'),
                             'Matched format comparison is missing')
        return importlib.import_module('research.structured_compare')

    def test_invalid_format_cannot_be_counted_as_a_functional_pass(self):
        module = self.module()
        task = {'id': 'fixture', 'editable': ['solution.py']}
        calls = []
        def grade(task, patch):
            calls.append(patch)
            return {'passed': True}
        for reply in [response('{"solution.py":"pass"}', 'length'),
                      response('{"wrong.py":"pass"}'), response('{"solution.py":"def"}')]:
            row = module.evaluate(task, reply, 1, grade)
            self.assertFalse(row['functional_passed'])
        self.assertEqual(calls, [])
        row = module.evaluate(task, response('{"solution.py":"pass"}'), 1, grade)
        self.assertTrue(row['functional_passed'])
        self.assertEqual(len(calls), 1)

    def test_incomplete_pairs_are_not_a_result_and_losses_stay_visible(self):
        module = self.module()
        def row(name, mode, passed):
            return dict(id=name, mode=mode, format_valid=True, syntax_valid=True,
                        functional_passed=passed, elapsed_s=1)
        rows = [row('a', 'plain', True), row('a', 'constrained', False),
                row('b', 'plain', False), row('b', 'constrained', True)]
        for invalid in [rows[:3], rows + [rows[0]], rows[:2] + [row('x', 'plain', False), rows[-1]]]:
            with self.assertRaises(ValueError):
                module.summarize(invalid, ['a', 'b'])
        result = module.summarize(rows, ['a', 'b'])
        self.assertEqual(result['functional_gains'], ['b'])
        self.assertEqual(result['functional_losses'], ['a'])
        self.assertFalse(result['learning_gain_established'])
        self.assertFalse(result['full_loop_authorized'])


if __name__ == '__main__':
    unittest.main()
