import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import desktop_bridge as app

class DesktopBridgeTests(unittest.TestCase):
    def test_session_path_cannot_escape_private_folder(self):
        for name in ('../escape', '/tmp/x', '.cache/learning/../x', 'a/b', ''):
            with self.assertRaises(ValueError): app.session_path(name)

    def test_partial_results_never_claim_improvement(self):
        self.assertEqual(app.verdict(None)['label'], 'Not yet measured')
        self.assertEqual(app.verdict({'completed': False})['label'], 'Not yet measured')

    def test_more_passes_with_losses_is_mixed(self):
        report={'completed':True,'comparisons':{'training_effect':{'audit':{'go':{
            'before':3,'after':4,'total':5,'gained':['a','b'],'lost':['c']}}}}}
        self.assertEqual(app.verdict(report)['label'], 'Mixed results')
        report['comparisons']['training_effect']['audit']['go']['lost']=[]
        self.assertEqual(app.verdict(report)['label'], 'Improved on these tests')
        report['comparisons']['training_effect']['audit']['go'].update(after=2,lost=['c'])
        self.assertEqual(app.verdict(report)['label'], 'Regressed on these tests')

    def test_unknown_action_cannot_be_a_command(self):
        with self.assertRaises(ValueError): app.dispatch({'action':'sh'})

    def test_log_tail_is_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'log'; p.write_text('x'*100000+'last line')
            value=app.tail(p)
            self.assertLessEqual(len(value),24000)
            self.assertTrue(value.endswith('last line'))

    def test_final_report_is_ignored_when_inputs_change(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'row.json').write_text('changed')
            (p/'comparison.json').write_text(json.dumps({'completed':True,'inputs':{'row.json':'0'*64}}))
            with self.assertRaises(ValueError): app.checked_report(p)

    def test_report_is_provisional_until_controller_binds_it(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'comparison.json').write_text(json.dumps({'completed':True}))
            (p/'status.json').write_text(json.dumps({'completed_stages':[]}))
            self.assertIsNone(app.checked_report(p))
            (p/'status.json').write_text(json.dumps({'completed_stages':[{'result':'comparison.json','sha256':'0'*64}]}))
            with self.assertRaisesRegex(ValueError,'comparison changed'): app.checked_report(p)

    def test_new_experiment_requires_valid_name_and_budget(self):
        for request in ({'name':'../bad','hours':1},{'name':'trial','hours':24}):
            with self.assertRaises(ValueError): app.prepare(request)


class RecipeTests(unittest.TestCase):
    def test_preparation_copies_private_data_and_pins_revision(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); data=root/'input.jsonl'; data.write_text('{}\n')
            Path(str(data)+'.manifest.json').write_text('{}')
            for name in ('focused_experiment.py','focused_evaluation.py','qwen35_train.py','qwen35_server.py',
                         'desktop_bridge.py','focused_report.py','focused_data.py','focused_replay.py','train_adapter.py','training_data.py','evaluation.py',
                         'launcher.py','polyglot_runtime.py','research/polyglot_benchmark.py',
                         'research/adapter_eval_server.py','.cache/rlm-env/bin/python',
                         '.cache/qwen35-env/bin/python','.cache/polyglot-runtime.json'):
                p=root/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('test')
            with patch.object(app,'ROOT',root),patch.object(app,'SESSIONS',root/'.cache/learning'),patch.object(app,'dataset_info',return_value={'rows':2000}):
                result=app.prepare({'name':'trial','hours':3,'dataset':str(data)})
                session=root/'.cache/learning/trial'
                config=json.loads((session/'config.json').read_text())
                self.assertEqual(result['status'],'paused')
                self.assertEqual(config['limit_seconds'],10800)
                self.assertEqual(len(config['stages']),4)
                command=config['stages'][1]['command']
                self.assertEqual(command[command.index('--revision')+1],app.REVISION)
                self.assertEqual((session/'data/train.jsonl').read_bytes(),data.read_bytes())
                self.assertEqual(json.loads((session/'command.json').read_text())['action'],'pause')
                with self.assertRaises(ValueError):app.prepare({'name':'trial','hours':3,'dataset':str(data)})
                self.assertTrue((session/'config.json').exists())

    def test_mismatched_benchmark_settings_cannot_be_improvement(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);session=root/'trial';session.mkdir()
            for model in ('qwen35-base','qwen35-trained'):
                folder=session/(model+'-audit');folder.mkdir()
                (folder/'go.json').write_text(json.dumps([{'id':'one','passed':True}]))
                summary={'completed':True,'binding':{'temperature':0 if model.endswith('base') else 1},
                         'languages':{'go':{'passed':1,'total':1}},'passed':1,'total':1,'macro_pass_at_1':1}
                (folder/'summary.json').write_text(json.dumps(summary))
            with patch.object(app,'SESSIONS',root):
                with self.assertRaisesRegex(ValueError,'conditions differ'):app.summarize('trial')
                self.assertFalse((session/'comparison.json').exists())

if __name__=='__main__': unittest.main()
