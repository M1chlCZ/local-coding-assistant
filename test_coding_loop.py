import json
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import unittest
import contextlib
from unittest.mock import patch

from coding_loop import CodingLoop, promotion
from coding_loop_stage import matched_tasks, prepare
from focused_experiment import Experiment, open_experiment
from qwen35_train import atomic_json, sha256
from test_qwen35_training import row, data_files
import qwen35_train
import desktop_bridge


class LoopTests(unittest.TestCase):
    def test_promotion_rejects_partial_ties_and_regressions(self):
        old=[{'id':'one','passed':True},{'id':'two','passed':False}]
        self.assertFalse(promotion(old,old)['promising'])
        self.assertTrue(promotion(old,[dict(r,passed=True) for r in old])['promising'])
        self.assertFalse(promotion(old,[{'id':'one','passed':False},{'id':'two','passed':True}])['promising'])
        for rows in (old[:1],[dict(r,error='runtime') for r in old]):
            with self.assertRaises(ValueError):promotion(old,rows)

    def test_split_guard_uses_problem_family_across_languages(self):
        tasks=[{'id':'a-go','repository':'problem-a','split':'train'},
               {'id':'a-ts','repository':'problem-a','split':'dev'}]
        with self.assertRaises(ValueError):matched_tasks(tasks)
        with self.assertRaises(ValueError):matched_tasks([tasks[0],tasks[0]])

    def create(self, root):
        path=root/'loop'
        with patch('coding_loop.SOURCES',()):loop=CodingLoop.prepare(path)
        return loop

    def test_unlimited_cycle_retains_clock_and_durable_controls(self):
        with tempfile.TemporaryDirectory() as d:
            loop=self.create(Path(d));loop.save(active_seconds=90000,training_seconds=86400)
            self.assertIsNone(loop.state['limit_seconds'])
            self.assertIsInstance(open_experiment(loop.path),CodingLoop)
            loop.control('resume');loop.control('pause')
            self.assertEqual(open_experiment(loop.path).desired(),'pause')
            atomic_json(loop.path/'round-000001/finish.json',dict(completed=True,accepted=False,retained_adapter=None,decision='Rejected'))
            self.assertTrue(loop.next_cycle())
            other=open_experiment(loop.path)
            self.assertEqual(other.state['cycle'],2)
            self.assertEqual(other.state['training_seconds'],86400)
            self.assertEqual(other.desired(),'pause')
            self.assertTrue(other.config['stages'][0]['name'].startswith('round-000002/'))
            loop.save(status='stopped')
            with self.assertRaises(ValueError):loop.control('resume')

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux worker lifecycle')
    def test_real_worker_crosses_day_budget_and_pauses_after_second_cycle(self):
        with tempfile.TemporaryDirectory() as d:
            loop=self.create(Path(d));loop.save(training_seconds=90000)
            loop.control('resume')
            script = """
import sys
from pathlib import Path
import coding_loop
from qwen35_train import atomic_json
path=Path(sys.argv[1])
def fixture(path,cycle):
    folder=path/f'round-{cycle:06d}'
    body='from pathlib import Path; import json,time; p=Path('+repr(str(path))+'); '
    body+='(p/'+repr(f'round-{cycle:06d}/finish.json')+').write_text('+repr('{"completed":true,"accepted":false,"retained_adapter":null,"decision":"Fixture rejected"}')+'); '
    if cycle==2:body+='(p/"command.json").write_text(json.dumps(dict(action="pause",requested_at=time.time())))'
    return [dict(name=f'round-{cycle:06d}/finish',command=[sys.executable,'-c',body],result=f'round-{cycle:06d}/finish.json',gpu=False,train=True)]
coding_loop.stages=fixture
coding_loop.CodingLoop(path).run()
"""
            child=subprocess.Popen([sys.executable,'-c',script,str(loop.path)])
            try:
                deadline=time.monotonic()+20
                while time.monotonic()<deadline:
                    state=json.loads((loop.path/'status.json').read_text())
                    if state['status']=='paused' and state.get('cycle')==2:break
                    if child.poll() is not None:self.fail('Worker exited before user pause')
                    time.sleep(.1)
                self.assertEqual(state['status'],'paused')
                self.assertEqual(state['cycle'],2)
                self.assertGreaterEqual(state['training_seconds'],90000)
                self.assertEqual(json.loads((loop.path/'command.json').read_text())['action'],'pause')
            finally:
                child.terminate();child.wait(timeout=10)

    def test_finite_sessions_still_reject_missing_budget(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):Experiment.create(Path(d),dict(inputs={},limit_seconds=None,stages=[dict(name='x',command=['true'],result='x.json')]))

    def test_resume_reuses_preassigned_tasks_without_consuming_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'round-000001';folder.mkdir()
            atomic_json(folder/'tasks.json',dict(train=[dict(id='train',repository='a',split='train')],
                validation=[dict(id='dev',repository='b',split='dev')]))
            with patch('polyglot_data.prepare_batch',side_effect=AssertionError('Must not fetch')):
                result=prepare(root,folder,{})
            self.assertEqual(result['tasks_sha256'],sha256(folder/'tasks.json'))

    def test_local_verified_source_all_languages_and_initial_adapter_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=[row(l,l) for l in ('python','go','typescript','rust','dart')]
            for r in rows:r['verification']={'kind':'local_execution','passed':True}
            data,manifest=data_files(root,rows)
            m=json.loads(manifest.read_text());m['verification']='local_execution';atomic_json(manifest,m)
            self.assertEqual(len(qwen35_train.load_verified(data,manifest)[0]),5)
            args=qwen35_train.parser().parse_args(['--dataset',str(data),'--manifest',str(manifest),
                '--output',str(root/'out'),'--revision','a'*40,'--initial-adapter',str(root/'missing')])
            with self.assertRaises(FileNotFoundError):qwen35_train.request_binding(args)

    def test_loop_report_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            loop=self.create(Path(d));folder=loop.path/'round-000001'
            evidence=folder/'gate.json';atomic_json(evidence,{'completed':True})
            report=folder/'finish.json';atomic_json(report,{'completed':True,'inputs':{'round-000001/gate.json':sha256(evidence)}})
            loop.save(last_report='round-000001/finish.json',last_report_sha256=sha256(report))
            self.assertTrue(desktop_bridge.checked_report(loop.path)['completed'])
            evidence.write_text('{}')
            with self.assertRaises(ValueError):desktop_bridge.checked_report(loop.path)

    def test_execution_filtered_sampling_never_changes_evaluation(self):
        from coding_loop_stage import collect
        task=dict(id='task', language='go', cases=[{'input':'hidden','output':'secret'}])
        for phase in ('collect','baseline'):
            with tempfile.TemporaryDirectory() as d:
                folder=Path(d);atomic_json(folder/'tasks.json',{'train':[task]})
                with patch('coding_loop_stage.server',return_value=contextlib.nullcontext()), \
                     patch('coding_loop_stage.messages',return_value=[{'role':'user','content':'public prompt'}]), \
                     patch('coding_loop_stage.visible_cases',return_value=[]), \
                     patch('coding_loop_stage.ask',side_effect=['bad','good']) as ask, \
                     patch('coding_loop_stage.check_program',side_effect=lambda lang,code,cases:dict(passed=not cases or code=='good',feedback='hidden failure')):
                    result=collect(folder,[task],None,phase)
                    self.assertEqual(result['rows'][0]['passed'],phase=='collect')
                    self.assertEqual(ask.call_count,2 if phase=='collect' else 1)
                    if phase=='collect':
                        self.assertEqual(ask.call_args.kwargs['temperature'],.7)
                        self.assertNotIn('secret',str(ask.call_args))
                    # Restart reuses the completed answer without generating another candidate.
                    self.assertEqual(collect(folder,[task],None,phase),result)

    def test_sampling_is_opt_in_and_validated(self):
        from qwen35_server import sampling
        self.assertEqual(sampling({})[0],{'do_sample':False})
        self.assertEqual(sampling({'temperature':.7,'seed':10})[1],10)
        for data in ({'temperature':float('nan')},{'temperature':-1},{'seed':-1},{'seed':True}):
            with self.assertRaises(ValueError):sampling(data)

    def test_invalid_model_output_is_a_failed_answer_not_a_retry_loop(self):
        from coding_loop_stage import ask
        for content in ('','```repl\nprint(1)\n```'):
            with patch('evaluation.request',return_value={'responses':[{'choices':[{'message':{'content':content}}]}]}):
                self.assertEqual(ask([{'role':'user','content':'code'}]),'')

    def test_validation_gain_cannot_promote_a_regressing_or_partial_audit(self):
        from coding_loop_stage import finish, LANGUAGES
        with tempfile.TemporaryDirectory() as d:
            session=Path(d);folder=session/'round-000001';folder.mkdir()
            atomic_json(folder/'gate.json',dict(completed=True,promising=True))
            for name,passed in (('baseline',False),('candidate',True)):
                atomic_json(folder/(name+'.json'),dict(rows=[dict(id='validation',language='go',passed=passed)]))
            for name in ('audit-parent','audit-candidate'):
                atomic_json(folder/(name+'.json'),dict(completed=True))
                languages={}
                for language in LANGUAGES:
                    rows=[dict(id=f'{language}-{i}',passed=not(name=='audit-candidate' and language=='go' and i==0)) for i in range(158)]
                    atomic_json(folder/name/(language+'.json'),rows)
                    languages[language]=dict(total=158,passed=sum(r['passed'] for r in rows))
                atomic_json(folder/name/'summary.json',dict(completed=True,total=790,
                    passed=sum(l['passed'] for l in languages.values()),languages=languages,macro_pass_at_1=0,binding={}))
            self.assertFalse(finish(session,folder,None)['accepted'])
            # Restoring the lost pass allows acceptance, but not a shortened row report.
            path=folder/'audit-candidate/go.json';rows=json.loads(path.read_text());rows[0]['passed']=True;atomic_json(path,rows)
            summary=folder/'audit-candidate/summary.json';value=json.loads(summary.read_text());value['passed']=790;value['languages']['go']['passed']=158;atomic_json(summary,value)
            self.assertTrue(finish(session,folder,None)['accepted'])
            atomic_json(path,rows[:-1])
            with self.assertRaises(ValueError):finish(session,folder,None)

if __name__=='__main__':unittest.main()
