import json
from pathlib import Path
import tempfile
import sys
import types
import unittest
import subprocess
from unittest.mock import patch

import teacher_data as teacher


def example(name='combine_items'):
    return dict(id='example', input=f'Write a function {name} that adds two numbers.',
        output=f'```python\ndef {name}(a, b):\n    return a + b\n```',
        unit_tests=json.dumps([f'assert {name}({i}, 2) == {i+2}' for i in range(5)]),
        tests_execution_status=json.dumps(['pass']*5), average_test_score='1')


class TeacherTests(unittest.TestCase):
    def test_successful_exit_cannot_skip_teacher_tests(self):
        value=dict(tests='assert answer() == 42\nassert False',test_imports='')
        def execute(language,program,cases=None):
            result=subprocess.run([sys.executable,'-I','-S','-c',program],capture_output=True,text=True)
            return dict(passed=result.returncode==0 and (cases is None or
                        result.stdout.split()==cases[0]['output'].split()))
        with patch.object(teacher,'check_program',side_effect=execute):
            for answer in ('def answer(): raise SystemExit(0)',
                           'def answer():\n import os\n os._exit(0)',
                           'def answer(): return 42'):
                self.assertFalse(teacher.check(value,answer)['passed'])
            self.assertTrue(teacher.check(dict(value,tests='assert answer() == 42'),
                                         'def answer(): return 42')['passed'])

    def test_reference_and_tests_never_enter_validation_prompt(self):
        entry=teacher.convert(example(), set())
        task=teacher.task(entry, 'dev')
        self.assertNotIn('answer', task)
        prompt=json.dumps(teacher.messages(task))
        self.assertNotIn('return a + b', prompt)
        self.assertNotIn('assert ', prompt)

    def test_bad_unchecked_or_benchmark_overlapping_examples_are_rejected(self):
        self.assertIsNone(teacher.convert(dict(example(), average_test_score='.9'), set()))
        self.assertIsNone(teacher.convert(dict(example(), unit_tests='[]'), set()))
        self.assertIsNone(teacher.convert(dict(example(), unit_tests=json.dumps(['assert True']*5)), set()))
        self.assertIsNone(teacher.convert(example(), set(), {'combine_items'}))
        entry=teacher.convert(example(), set())
        self.assertIsNone(teacher.convert(example(), teacher.shingles(example()['input']+' '+entry['answer'])))

    def test_function_family_keeps_variants_in_one_split(self):
        a=teacher.convert(example(), set())
        b=teacher.convert(dict(example(),id='variant',input='Another description'), set())
        self.assertEqual(a['family'],b['family'])
        self.assertEqual(teacher.split_for(a),teacher.split_for(b))

    def test_reference_must_pass_and_empty_solution_must_fail(self):
        entry=teacher.convert(example(), set())
        with patch.object(teacher, 'check_program',side_effect=[{'passed':True},{'passed':False}]) as check:
            self.assertTrue(teacher.verify(entry))
            self.assertIn('assert combine_items',check.call_args_list[0].args[1])
        with patch.object(teacher, 'check_program',return_value={'passed':True}):
            self.assertFalse(teacher.verify(entry))

    def test_saved_teacher_evidence_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            entry=teacher.convert(example(), set())
            teacher.atomic_json(folder/'teacher.json',dict(train=[entry],validation=[]))
            teacher.atomic_json(folder/'prepare.json',dict(teacher_sha256=teacher.sha256(folder/'teacher.json')))
            self.assertEqual(teacher.load_round(folder)['train'],[entry])
            teacher.atomic_json(folder/'teacher.json',dict(train=[],validation=[]))
            with self.assertRaisesRegex(ValueError,'changed'):
                teacher.load_round(folder)

    def test_teacher_recipe_does_not_inherit_backoff_from_old_self_training(self):
        import coding_loop_stage as stage
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'round-000016';folder.mkdir()
            teacher.atomic_json(folder/'teacher.json',dict(train=[{}],validation=[]))
            cfg=dict(steps_per_round=128,max_epochs=1,learning_rate=2e-5,teacher_learning_rate=2e-5)
            recipe=stage.training_recipe(Path(d),folder,cfg)
            self.assertEqual(recipe.get('method'),'teacher_mix')
            self.assertEqual(recipe['learning_rate'],2e-5)
            teacher.atomic_json(folder/'adapter/request.json',dict(max_steps=53,max_epochs=1,learning_rate=5e-6))
            self.assertEqual(stage.training_recipe(Path(d),folder,cfg)['learning_rate'],5e-6)

    def test_preparation_resumes_and_uses_new_families_next_round(self):
        rows=[dict(example('function_'+str(i)),id=str(i)) for i in range(100)]
        parquet=types.SimpleNamespace(ParquetFile=lambda _:types.SimpleNamespace(
            iter_batches=lambda **_:iter([types.SimpleNamespace(to_pylist=lambda:rows)])))
        with tempfile.TemporaryDirectory() as d, \
             patch.dict(sys.modules,{'pyarrow.parquet':parquet}), \
             patch.object(teacher,'download',return_value=Path(d)/'source'), \
             patch.object(teacher,'benchmark_exclusions',return_value=(set(),{})), \
             patch('research.polyglot_benchmark.load_tasks',return_value=({},{})), \
             patch.object(teacher,'image',return_value='fixture'), \
             patch.object(teacher,'verify',return_value=True) as verify:
            session=Path(d);folder=session/'round-000001';folder.mkdir()
            first=teacher.prepare_round(session,folder,2,1,lambda *a,**k:None)
            calls=verify.call_count
            resumed=teacher.prepare_round(session,folder,2,1,lambda *a,**k:None)
            self.assertEqual(first,resumed)
            self.assertEqual(verify.call_count,calls)
            teacher.atomic_json(folder/'prepare.json',dict(teacher_sha256=teacher.sha256(folder/'teacher.json')))
            folder2=session/'round-000002';folder2.mkdir()
            second=teacher.prepare_round(session,folder2,2,1,lambda *a,**k:None)
            self.assertFalse(set(first['families']) & set(second['families']))
            self.assertTrue(all(teacher.split_for(e)=='train' for e in second['train']))
            self.assertTrue(all(teacher.split_for(e)=='dev' for e in second['validation']))

    def test_function_checks_and_prompts_are_routed_without_stdin_cases(self):
        import coding_loop_stage as stage
        value=teacher.task(teacher.convert(example(),set()),'dev')
        with patch.object(teacher,'check',return_value={'passed':True}) as check:
            self.assertTrue(stage.check_task(value,'answer')['passed'])
            check.assert_called_once_with(value,'answer')
        self.assertEqual(stage.task_messages(value),teacher.messages(value))

    def test_test_imports_do_not_depend_on_candidate_import_style(self):
        raw=example('area')
        raw.update(output='import math\ndef area(r):\n    return math.pi*r*r',
                   unit_tests=json.dumps(['assert area(1) == math.pi']*5))
        value=teacher.task(teacher.convert(raw,set()),'dev')
        answer='from math import pi\ndef area(r):\n    return pi*r*r'
        with patch.object(teacher,'check_program') as check:
            teacher.check(value,answer)
            exec(check.call_args.args[1], {})
            teacher.check(value,'def area(r):\n    return math.pi*r*r')
            with self.assertRaises(NameError):exec(check.call_args.args[1], {})

    def test_teacher_validation_regression_adjusts_the_next_unstarted_recipe(self):
        import coding_loop_stage as stage
        with tempfile.TemporaryDirectory() as d:
            session=Path(d);old=session/'round-000016';folder=session/'round-000017'
            teacher.atomic_json(old/'teacher.json',{})
            teacher.atomic_json(folder/'teacher.json',{})
            run=old/'adapter/run.json';teacher.atomic_json(run,dict(learning_rate=2e-5))
            teacher.atomic_json(old/'finish.json',dict(completed=True,
                comparisons=dict(training_effect=dict(validation=dict(python=dict(lost=['x'])))),
                inputs={str(run.relative_to(session)):teacher.sha256(run)}))
            recipe=stage.training_recipe(session,folder,dict(steps_per_round=128,learning_rate=2e-5))
            self.assertEqual(recipe['learning_rate'],1e-5)
            self.assertEqual(recipe['previous_evidence_suite'],'validation')

    def test_teacher_audit_after_three_updates_even_when_validation_rejects(self):
        import coding_loop_stage as stage
        with tempfile.TemporaryDirectory() as d:
            session=Path(d)
            for i in range(16,19):
                f=session/f'round-{i:06d}'
                teacher.atomic_json(f/'teacher.json',{})
                teacher.atomic_json(f/'adapter/training.json',dict(status='completed'))
            gate=dict(promising=False)
            self.assertFalse(stage.audit_due(session,session/'round-000017',gate,{}))
            self.assertTrue(stage.audit_due(session,session/'round-000018',gate,{}))
            teacher.atomic_json(session/'round-000018/finish.json',dict(completed=True,
                comparisons=dict(training_effect=dict(audit={'python':{}}))))
            f=session/'round-000019';teacher.atomic_json(f/'teacher.json',{})
            teacher.atomic_json(f/'adapter/training.json',dict(status='completed'))
            self.assertFalse(stage.audit_due(session,f,gate,{}))


if __name__ == '__main__': unittest.main()
