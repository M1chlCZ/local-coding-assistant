import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import desktop_bridge as app

class DesktopBridgeTests(unittest.TestCase):
    def test_session_path_cannot_escape_private_folder(self):
        for name in ('../escape', '/tmp/x', '.cache/learning/../x', 'a/b', ''):
            with self.assertRaises(ValueError): app.session_path(name)

    def test_task_preview_contains_only_current_editable_code(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'round-000001';folder.mkdir()
            task=dict(id='current',filename='main.py',language='python',editable=['main.py'],
                      files={'main.py':'print(1)','hidden.py':'SECRET'},checks='SECRET')
            (folder/'tasks.json').write_text(json.dumps(dict(train=[task],validation=[])))
            state=dict(continuous=True,cycle=1,stage_progress=dict(task='current'),phase='round-000001/collect')
            preview=app.task_preview(Path(d),state)
            self.assertEqual(preview['code'],'print(1)')
            self.assertNotIn('SECRET',str(preview))
            (folder/'collect-attempt.json').write_text(json.dumps(dict(id='different',answer='wrong task')))
            self.assertEqual(app.task_preview(Path(d),state)['code'],'print(1)')
            (folder/'collect-attempt.json').write_text(json.dumps(dict(id='current',answer='print(2)')))
            self.assertEqual(app.task_preview(Path(d),state)['code'],'print(2)')
            state['stage_progress']['task']='unknown'
            self.assertIsNone(app.task_preview(Path(d),state))

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
            for name in ('focused_experiment.py','focused_evaluation.py','qwen35_train.py','qwen35_server.py','structured_output.py',
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


class AgentCurriculumStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root/'pilot'
        self.path.mkdir()
        self.config = dict(learning_method='agent_curriculum', inputs={}, stages=[
            dict(name='train-agent', result='candidate-1/adapter/training.json', train=True),
            dict(name='compare-agent', result='candidate-1/development.json')])
        self.state = dict(continuous=False, phase='compare-agent', stage_index=1, status='running',
                          stage_progress=dict(phase='load-model', completed=0, total=0, heartbeat_at=1))
        rows = [dict(task_id=str(i), split='train', language='python',
                     example_kind='executed_repl_repair' if i < 2 else 'complete_code',
                     messages=[dict(role='user', content='Task'), dict(role='assistant', content='Action')],
                     verification=dict(kind='local_execution', passed=True),
                     provenance=dict(dataset='reference', revision='a'*40, source_row_sha256='b'*64))
                for i in range(3)]
        dataset = self.path/'curriculum/train.jsonl'
        dataset.parent.mkdir()
        dataset.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.bind(dataset)
        manifest = self.write('curriculum/train.jsonl.manifest.json', dict(
            schema_version=1, split='train', verification='local_execution', rows=3,
            languages=dict(python=3), dataset_sha256=app.sha256(dataset)), frozen=True)
        self.write('curriculum/feasibility.json', dict(passed=True, all_agent_actions_included=True,
            complete_trajectories=True, dataset_sha256=app.sha256(dataset),
            manifest_sha256=app.sha256(manifest)), frozen=True)
        self.entries = [dict(condition=condition, task=dict(id=identity, editable=['solver.py'],
            files={'solver.py':source, 'private.py':'SECRET'}, checks='SECRET', reference_patch='SECRET'))
            for condition, identity, source in [('ordinary', 'first', 'return 1'), ('recursive', 'next', 'return 2')]]
        self.write('evaluation.json', dict(development=self.entries, holdout=[], retention=[dict(id='anchor')]), frozen=True)
        self.write('status.json', dict(completed_stages=[]))

    def bind(self, file):
        self.config['inputs'][str(file)] = app.sha256(file)

    def write(self, name, value, frozen=False):
        file = self.path/name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(value))
        if frozen:
            self.bind(file)
        return file

    def results(self, completed=1, retention=0):
        value = dict(binding=dict(evaluation_sha256=app.sha256(self.path/'evaluation.json'), suite='development'),
            rows=[dict(id=e['task']['id'], condition=e['condition'], passed=True) for e in self.entries[:completed]],
            retention=[dict(id='anchor', passed=True)] if retention else [])
        self.write('candidate-1/development.json', value)
        return value

    def snapshot(self):
        experiment = SimpleNamespace(config=self.config, snapshot=lambda:dict(self.state))
        with patch.object(app, 'SESSIONS', self.root), patch.object(app, 'open_experiment', return_value=experiment), \
                patch.object(app.subprocess, 'run', side_effect=OSError):
            return app.snapshot('pilot')

    def test_verified_scripted_counts_preserve_training_progress(self):
        self.state.update(phase='train-agent', stage_index=0, stage_progress=dict(phase='train', step=10, max_steps=200))
        before = {p:p.read_bytes() for p in self.path.rglob('*') if p.is_file()}
        value = self.snapshot()
        self.assertEqual((value['repl_examples'], value['training_examples']), (2, 3))
        self.assertIn('verified scripted REPL training rows', value['readiness'])
        self.assertEqual(value['stage_progress']['step'], 10)
        for name in ('collected_examples', 'first_passes', 'repaired_examples', 'accepted_updates'):
            self.assertNotIn(name, value)
        self.assertEqual(before, {p:p.read_bytes() for p in self.path.rglob('*') if p.is_file()})

    def test_collection_is_unscored_and_completed_collection_binds_its_dataset(self):
        self.config['stages'].insert(0, dict(name='collect', result='curriculum/feasibility.json'))
        for file in (self.path/'curriculum').iterdir():
            self.config['inputs'].pop(str(file))
        self.state.update(phase='collect', stage_index=0)
        value = self.snapshot()
        self.assertEqual(value['quality']['label'], 'Not yet measured')
        self.assertNotIn('training_examples', value)
        self.assertIn('Collecting', value['readiness'])
        self.state['phase']='prepare'
        value=self.snapshot()
        self.assertEqual(value['quality']['label'],'Not yet measured')
        self.assertNotIn('training_examples',value)
        self.assertIn('Preparing',value['readiness'])
        evidence = self.path/'curriculum/feasibility.json'
        self.state.update(phase='train-agent', stage_index=1, completed_stages=[
            dict(name='collect', result='curriculum/feasibility.json', sha256=app.sha256(evidence))])
        self.assertEqual(self.snapshot()['training_examples'], 3)
        for file in (self.path/'curriculum').iterdir():
            original = file.read_bytes()
            file.write_bytes(original+b' ')
            with self.subTest(file=file.name), self.assertRaisesRegex(ValueError, 'display evidence changed'):
                self.snapshot()
            file.write_bytes(original)

    def test_saved_progress_uses_only_current_frozen_public_source_without_scoring(self):
        self.results()
        mutable = json.loads(json.dumps(self.entries[1]))
        mutable['task']['files']['solver.py'] = 'SECRET mutable source'
        self.write('candidate-1/task.json', mutable)
        self.write('comparison.json', dict(completed=True))  # Controller has not bound a final report.
        value = self.snapshot()
        self.assertEqual(value['stage_progress'], dict(phase='repl-candidate', completed=1, total=3, task='next'))
        self.assertEqual(value['task_preview']['code'], 'return 2')
        self.assertNotIn('SECRET', str(value['task_preview']))
        self.assertIsNone(value['report'])
        self.assertEqual(value['quality']['label'], 'Not yet measured')
        self.assertEqual(value['quality']['total'], 0)

    def test_completed_pilot_shows_only_the_bound_decision(self):
        self.state.update(status='completed', stage_index=2, phase='completed')
        report = dict(completed=True, reason='Pilot gain requirement not met',
                      inputs={'evaluation.json':app.sha256(self.path/'evaluation.json')},
                      comparisons=dict(training_effect={}))
        file = self.write('comparison.json', report)
        self.assertIn('Waiting for a complete comparison', self.snapshot()['readiness'])
        self.write('status.json', dict(completed_stages=[dict(result='comparison.json', sha256=app.sha256(file))]))
        self.assertEqual(self.snapshot()['readiness'], report['reason'])
        report['reason'] = 'Unbound result'
        self.write('comparison.json', report)
        with self.assertRaisesRegex(ValueError, 'Completed comparison changed'):
            self.snapshot()

    def test_compact_pilot_gate_has_bound_evidence_and_measured_counts(self):
        self.state.update(status='completed', stage_index=2, phase='completed')
        gate = dict(completed=True, before=0, after=2, total=10, gained_families=['a','b'], lost=[],
            reason='Confirmation still required', inputs={'evaluation.json':app.sha256(self.path/'evaluation.json')})
        file = self.write('comparison.json', dict(completed=True, pilot_gate=gate, full_loop_authorized=False))
        self.write('status.json', dict(completed_stages=[dict(result='comparison.json',sha256=app.sha256(file))]))
        value = self.snapshot()
        self.assertEqual((value['quality']['before'],value['quality']['after'],value['quality']['total']), (0,2,10))
        self.assertEqual(value['readiness'], gate['reason'])
        self.assertFalse(value['report']['full_loop_authorized'])
        self.write('evaluation.json', dict(changed=True))
        with self.assertRaisesRegex(ValueError, 'Comparison evidence changed'):
            self.snapshot()

    def test_finite_pilot_lists_checkpoints_from_its_training_stage(self):
        marker = self.write('candidate-1/adapter/checkpoint-25/complete.json', dict(completed=True))
        self.assertEqual(self.snapshot()['checkpoints'], [dict(
            path='candidate-1/adapter/checkpoint-25', saved_at=marker.stat().st_mtime)])

    def test_loading_is_preserved_until_first_task_but_first_task_shows_zero_completed(self):
        self.assertEqual(self.snapshot()['stage_progress']['phase'], 'load-model')
        self.write('candidate-1/task.json', self.entries[0])
        value = self.snapshot()
        self.assertEqual(value['stage_progress']['completed'], 0)
        self.assertEqual(value['stage_progress']['total'], 3)
        self.assertEqual(value['task_preview']['code'], 'return 1')

    def test_stale_task_source_is_hidden_during_transition_and_retention(self):
        self.results()
        self.write('candidate-1/task.json', self.entries[0])
        self.assertIsNone(self.snapshot()['task_preview'])
        self.results(completed=2)
        self.write('candidate-1/task.json', self.entries[1])
        value = self.snapshot()
        self.assertIsNone(value['task_preview'])
        self.assertEqual(value['stage_progress']['task'], 'Retention checks')
        self.assertEqual(value['stage_progress']['completed'], 2)
        self.results(completed=2, retention=1)
        value = self.snapshot()
        self.assertEqual(value['stage_progress']['completed'], 3)
        self.assertEqual(value['stage_progress']['task'], 'Comparison tasks completed')
        self.assertEqual(value['quality']['total'], 0)

    def test_changed_frozen_curriculum_is_rejected(self):
        for name in ('curriculum/train.jsonl', 'curriculum/train.jsonl.manifest.json', 'curriculum/feasibility.json'):
            file = self.path/name
            original = file.read_bytes()
            file.write_bytes(original+b' ')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'display evidence changed'):
                self.snapshot()
            file.write_bytes(original)

    def test_explicitly_bound_shared_curriculum_and_baseline_progress(self):
        shared = self.root/'verified-curriculum'
        (self.path/'curriculum').rename(shared)
        (self.path/'curriculum').symlink_to(shared, target_is_directory=True)
        self.config['stages'][1]['result'] = 'baseline/development.json'
        self.write('baseline/development.json', self.results())
        self.write('baseline/task.json', self.entries[1])
        value = self.snapshot()
        self.assertEqual((value['repl_examples'], value['training_examples']), (2, 3))
        self.assertEqual(value['stage_progress']['phase'], 'repl-baseline')
        self.assertEqual(value['task_preview']['code'], 'return 2')
        del self.config['inputs'][str(self.path/'curriculum/train.jsonl')]
        with self.assertRaisesRegex(ValueError, 'display evidence changed'):
            self.snapshot()

    def test_mismatched_or_unfinished_results_cannot_inflate_completed_count(self):
        for change in ('binding', 'order', 'extra', 'retention', 'infrastructure'):
            value = self.results()
            if change == 'binding': value['binding']['evaluation_sha256'] = '0'*64
            if change == 'order': value['rows'][0]['id'] = 'next'
            if change == 'extra': value['rows'] *= 3
            if change == 'retention': value['retention'] = [dict(id='anchor', passed=True)]
            if change == 'infrastructure': value['rows'][0]['infrastructure_errors'] = ['Service unavailable']
            self.write('candidate-1/development.json', value)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'frozen comparison tasks'):
                self.snapshot()

    def test_other_recipes_and_outside_session_paths(self):
        self.config['learning_method'] = 'other'
        self.assertNotIn('repl_examples', self.snapshot())
        self.config['learning_method'] = 'agent_curriculum'
        self.config['stages'][1]['result'] = '../outside/development.json'
        with self.assertRaisesRegex(ValueError, 'stay in its session'):
            self.snapshot()

if __name__=='__main__': unittest.main()
