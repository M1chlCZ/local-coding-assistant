import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

class PolyglotTests(unittest.TestCase):
    def test_go_test_imports_are_present_when_answer_has_its_own_package(self):
        from research.polyglot_benchmark import program
        task={'prompt':'package main\nimport ("testing";"fmt")\nfunc answer() int {',
              'tests':'func TestAnswer(t *testing.T) { t.Error(fmt.Sprint(answer())) }'}
        for code in ('package main\nfunc answer() int {return 1}',
                     'package main\nimport "fmt"\nfunc answer() int {fmt.Println();return 1}'):
            source=program('go',task,code)
            self.assertEqual(source.count('"testing"'),1)
            self.assertEqual(source.count('"fmt"'),1)
            self.assertIn(code.split('\n',1)[1],source)

    def test_go_harness_does_not_supply_missing_algorithm_imports(self):
        from research.polyglot_benchmark import program
        task={'prompt':'package main\nimport ("testing";"fmt";"sort")\nfunc answer() int {',
              'tests':'func TestAnswer(t *testing.T) { t.Error(fmt.Sprint(answer())) }'}
        source=program('go',task,'package main\nfunc answer() int {sort.Ints(nil);return 1}')
        self.assertNotIn('"sort"',source)

    def test_native_runner_rejects_wrong_answer_and_empty_output(self):
        from polyglot_runtime import runner_source
        task={'language':'go','filename':'solution.go','cases':[{'input':'2 3\n','output':'5\n'}]}
        source=runner_source(task)
        self.assertIn("['go', 'build'",source)
        self.assertIn('output.split()',source)
        self.assertIn('raise AssertionError',source)

    def test_reference_cases_do_not_enter_prompt(self):
        from polyglot_data import tasks_from_row
        row={'name':'sum','description':'Read two integers and print their sum.',
             'public_tests':{'input':['2 3\n'],'output':['5\n']},
             'private_tests':{'input':['1 7\n','0 0\n'],'output':['8\n','0\n']},
             'generated_tests':{'input':[],'output':[]},
             'solutions':{'language':[3],'solution':['a,b=map(int,input().split());print(a+b)']},
             'input_file':'','output_file':''}
        tasks=tasks_from_row(row)
        self.assertEqual([t['language'] for t in tasks],['python','go','typescript','rust','dart'])
        self.assertEqual(len({t['split'] for t in tasks}),1)
        self.assertEqual(len({t['repository'] for t in tasks}),1)
        self.assertTrue(all('1 7' not in t['prompt'] and '1 7' not in t['files']['test_visible.py'] for t in tasks))
        self.assertTrue(all(len(t['cases'])==3 for t in tasks))

    def test_rotation_is_balanced_and_covers_five_languages(self):
        from polyglot_data import rotate
        tasks=[{'id':f'{l}-{i}','language':l,'split':'train'} for l in ['python','go','typescript','rust','dart'] for i in range(16)]
        ordered=rotate(tasks)
        self.assertEqual(len(ordered),80)
        self.assertEqual(len({t['id'] for t in ordered}),80)
        for index,l in enumerate(['python','go','typescript','rust','dart']):
            batch=ordered[index*16:(index+1)*16]
            self.assertEqual(batch[0]['language'],l)
            self.assertEqual({t['language'] for t in batch},{'python','go','typescript','rust','dart'})

    def test_all_languages_reported_without_hiding_partial_checks(self):
        from polyglot_session import language_scores
        tasks=[{'id':'p','language':'python'},{'id':'g','language':'go'},{'id':'d','language':'dart'}]
        result=language_scores(tasks,{'tasks':[{'id':'p','passed':True},{'id':'g','passed':False}]})
        self.assertTrue(result['python']['complete']);self.assertFalse(result['dart']['complete'])
        self.assertEqual(result['go']['passed'],0)

    def test_standardized_audit_requires_complete_every_language(self):
        from research.polyglot_benchmark import summarize
        tasks={'python':[{'id':'x'}],'go':[{'id':'y'}]}
        binding={'languages':['python','go']}
        reports={mode:{l:[{'id':t['id'],'passed':True,'elapsed_s':1}] for l,ts in tasks.items() for t in ts} for mode in ['base','adapter']}
        value=summarize(binding,tasks,reports)
        self.assertTrue(value['completed']);self.assertEqual(value['macro_pass_at_1']['adapter'],1)
        reports['adapter']['go']=[]
        with self.assertRaises(ValueError):summarize(binding,tasks,reports)


class ScheduledPolyglotTests(unittest.TestCase):
    def test_multilingual_audit_is_selected_and_keeps_separate_cache(self):
        from continuous_learning import Controller
        from train_adapter import atomic_json
        from training_data import sha256
        with tempfile.TemporaryDirectory() as root:
            path=Path(root);adapter=path/'adapter';adapter.mkdir()
            (adapter/'adapter_model.safetensors').write_text('weights')
            child=path/'child';atomic_json(child/'status.json',{'status':'completed','polyglot':True,
                'best_adapter':str(adapter),'confirmation':{'baseline_adapter':str(adapter)}})
            controller=Controller.create(path/'controller',child)
            atomic_json(controller.path/'confirmations'/child.name/'summary.json',{'completed':True,'regressions':[]})
            command,output=controller.audit_command()
            self.assertEqual(Path(command[1]).name,'polyglot_benchmark.py')
            self.assertEqual(output.parent.name,'polyglot-benchmarks')

    def test_continuation_replaces_clock_key_without_counting_budget_twice(self):
        from continuous_learning import Controller
        from train_adapter import atomic_json
        with tempfile.TemporaryDirectory() as root:
            path=Path(root);old=path/'old';new=path/'new'
            atomic_json(old/'status.json',{'status':'paused','active_seconds':27000})
            atomic_json(new/'status.json',{'status':'paused','active_seconds':27000})
            controller=Controller.create(path/'controller',old)
            controller.attach_continuation(new)
            self.assertEqual(controller.snapshot()['active_seconds'],27000)
            self.assertEqual(controller.state['continuations'][-1]['active_seconds_carried'],27000)


class PreparationTests(unittest.TestCase):
    def test_preparation_requires_idle_paused_worker(self):
        from polyglot_session import prepare
        from train_adapter import atomic_json
        from continuous_learning import Controller
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);child=root/'child'
            atomic_json(child/'status.json',{'status':'running','active_seconds':100,'limit_seconds':43200})
            atomic_json(child/'command.json',{'action':'resume'})
            controller=Controller.create(root/'controller',child)
            with self.assertRaisesRegex(ValueError,'Pause'):
                prepare(controller.path,root/'missing.json')

if __name__=='__main__':unittest.main()
