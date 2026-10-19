"""Checked correction conversations and separate REPL evaluation contracts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import coding_loop_stage as stage
from coding_loop import CodingLoop
from qwen35_train import atomic_json


def task():
    return dict(id='repair-go', repository='family', split='train', language='go',
                filename='solution.go', editable=['solution.go'], prompt='Repair the program',
                files={'solution.go': 'broken', 'test_visible.py': "cases=[{'input':'public','output':'ok'}]"},
                cases=[{'input':'hidden','output':'SECRET'}], provenance=dict(dataset='fixture',revision='a'*40))


class CorrectionTests(unittest.TestCase):
    def test_repairs_keep_only_latest_failure_in_model_context(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);t=task();tasks=[t,dict(t,id='second')];atomic_json(folder/'tasks.json',dict(train=tasks))
            answers=iter(['bad-1','bad-2','bad-3','fixed','fixed'])
            def ask(messages):
                self.assertLessEqual(len(messages),4,'Old full programs overflow the repair context')
                return next(answers)
            with patch.object(stage,'server',return_value=contextlib.nullcontext()), \
                 patch.object(stage,'ask',side_effect=ask), \
                 patch.object(stage,'check_program',side_effect=lambda *_args:dict(passed=_args[1]=='fixed',feedback='public failure')):
                row=stage.collect(folder,tasks,None,'collect',repair_turns=3)['rows'][0]
            self.assertTrue(row['passed'])
            self.assertEqual(row['repair_attempts'],3)
            self.assertEqual(row['repair_messages'][-3]['content'],'bad-3')
            self.assertNotIn('bad-1',str(row['repair_messages']))
            self.assertEqual(json.loads((folder/'collect-attempt.json').read_text())['id'],'second')
            saved=json.loads((folder/'collect-rows.json').read_text())['rows'][0]
            self.assertEqual(saved['attempts']['answer'],'bad-1')
            self.assertEqual(saved['attempts']['correction'],'bad-2')
            self.assertEqual(saved['attempts']['correction-1'],'bad-3')

    def test_model_request_rejections_keep_reason_and_classify_retry(self):
        for status,error_type in [(400,ValueError),(404,ValueError),(429,RuntimeError),(500,RuntimeError)]:
            error=HTTPError('http://localhost/batch',status,'failure',{},io.BytesIO(b'{"error":"context too long"}'))
            with patch('evaluation.request',side_effect=error):
                with self.assertRaisesRegex(error_type,f'HTTP {status}.*context too long'):
                    stage.ask([dict(role='user',content='code')])

    def test_multiple_repairs_keep_actual_failure_context_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);t=task();atomic_json(folder/'tasks.json',dict(train=[t]))
            def check(language, answer, cases):
                return dict(passed=answer=='fixed', feedback='public failure' if cases[0]['input']=='public' else 'SECRET')
            with patch.object(stage,'server',return_value=contextlib.nullcontext()), \
                 patch.object(stage,'ask',side_effect=['bad','still bad','fixed']) as ask, \
                 patch.object(stage,'check_program',side_effect=check):
                result=stage.collect(folder,[t],None,'collect',repair_turns=3)
                row=result['rows'][0]
                self.assertTrue(row['passed'])
                self.assertFalse(row['first_passed'])
                self.assertEqual(row['repair_attempts'],2)
                self.assertEqual(row['repair_messages'][-1],dict(role='assistant',content='fixed'))
                self.assertIn('still bad',str(row['repair_messages']))
                self.assertNotIn('SECRET',str(row['repair_messages']))
                self.assertEqual(stage.collect(folder,[t],None,'collect',repair_turns=3),result)
                self.assertEqual(ask.call_count,3)

    def test_hidden_failure_never_becomes_correction_feedback(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);t=task();atomic_json(folder/'tasks.json',dict(train=[t]))
            with patch.object(stage,'server',return_value=contextlib.nullcontext()), \
                 patch.object(stage,'ask',return_value='bad') as ask, \
                 patch.object(stage,'check_program',side_effect=lambda l,a,c:dict(passed=c[0]['input']=='public',feedback='SECRET')):
                row=stage.collect(folder,[t],None,'collect',repair_turns=3)['rows'][0]
                self.assertFalse(row['passed'])
                self.assertNotIn('repair_messages',row)
                self.assertNotIn('SECRET',str(ask.call_args_list))

    def test_evaluation_stays_single_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);t=task();atomic_json(folder/'tasks.json',dict(validation=[t]))
            with patch.object(stage,'server',return_value=contextlib.nullcontext()), \
                 patch.object(stage,'ask',return_value='bad') as ask, \
                 patch.object(stage,'check_program',return_value=dict(passed=False,feedback='SECRET')):
                result=stage.collect(folder,[t],None,'baseline',repair_turns=3)
                self.assertFalse(result['rows'][0]['passed'])
                self.assertEqual(ask.call_count,1)

    def test_repair_gate_requires_matched_repl_evidence_without_losses(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            direct=[dict(id='direct',passed=True)]
            for name in ('baseline','candidate'):atomic_json(folder/(name+'.json'),dict(rows=direct))
            atomic_json(folder/'repl-baseline.json',dict(rows=[dict(id='repl',passed=False)]))
            atomic_json(folder/'repl-candidate.json',dict(rows=[dict(id='repl',passed=True)]))
            self.assertTrue(stage.validation_gate(folder,dict(learning_method='execution_repair'))['promising'])
            atomic_json(folder/'candidate.json',dict(rows=[dict(id='direct',passed=False)]))
            self.assertFalse(stage.validation_gate(folder,dict(learning_method='execution_repair'))['promising'])
            (folder/'repl-candidate.json').unlink()
            with self.assertRaises(ValueError):stage.validation_gate(folder,dict(learning_method='execution_repair'))

    def test_new_method_is_bound_and_old_sessions_keep_original_stages(self):
        with tempfile.TemporaryDirectory() as d, patch('coding_loop.SOURCES',()):
            loop=CodingLoop.prepare(Path(d)/'loop')
            self.assertEqual(loop.config['learning_method'],'execution_repair')
            names=[s['name'].split('/')[-1] for s in loop.config['stages']]
            self.assertLess(names.index('repl-baseline'),names.index('train'))
            self.assertLess(names.index('repl-candidate'),names.index('gate'))
            self.assertEqual(loop.snapshot()['teaching_method'],'execution_repair')

    def test_repl_lessons_are_executed_train_only_and_have_unique_ids(self):
        import repl_learning
        t=task()
        calls=[([dict(role='system',content='REPL'),dict(role='user',content='public task')],
                '```repl\nprint(context)\n```'),
               ([dict(role='user',content='REPL output: public failure')],
                '```repl\nanswer["ready"] = True\n```')]
        with patch('recursive_agent.lesson_prompt',return_value=calls[0][0]), \
             patch('training_data.repair_calls',return_value=calls) as execute:
            rows=repl_learning.lesson_rows(t,dict(answer='fixed',initial_answer='bad',first_passed=False))
            self.assertEqual(len({r['task_id'] for r in rows}),2)
            self.assertTrue(all(r['example_kind']=='executed_repl_repair' for r in rows))
            self.assertNotIn('SECRET',str([r['messages'] for r in rows]))
            self.assertEqual(execute.call_args.args[0]['files']['solution.go'],'bad')
            with self.assertRaises(ValueError):repl_learning.lesson_rows(dict(t,split='dev'),dict(answer='fixed'))

    def test_lessons_that_do_not_reproduce_failure_are_excluded(self):
        import repl_learning
        with patch('recursive_agent.lesson_prompt',return_value=[]), \
             patch('training_data.repair_calls',side_effect=ValueError('Visible tests do not reproduce the defect')):
            self.assertEqual(repl_learning.lesson_rows(task(),dict(answer='fixed')),[])
        with patch('recursive_agent.lesson_prompt',return_value=[]), \
             patch('training_data.repair_calls',side_effect=RuntimeError('Docker unavailable')):
            with self.assertRaises(RuntimeError):repl_learning.lesson_rows(task(),dict(answer='fixed'))

    def test_repl_results_resume_only_matching_tasks_weights_and_limits(self):
        import repl_learning
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);t=task();atomic_json(folder/'tasks.json',dict(validation=[t]))
            config=dict(repl_tasks=5)
            with patch.object(stage,'server',return_value=contextlib.nullcontext()), \
                 patch.object(repl_learning,'run_task',return_value=dict(passed=True,calls=2,trace=[])) as solve:
                result=repl_learning.evaluate(folder,[t],None,'repl-baseline',config)
                self.assertEqual(repl_learning.evaluate(folder,[t],None,'repl-baseline',config),result)
                self.assertEqual(solve.call_count,1)
                atomic_json(folder/'tasks.json',dict(validation=[dict(t,prompt='changed')]))
                with self.assertRaises(ValueError):repl_learning.evaluate(folder,[t],None,'repl-baseline',config)

    def test_timed_out_generation_drains_before_the_next_task_budget(self):
        import repl_learning
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);tasks=[task(),dict(task(),id='second')]
            atomic_json(folder/'tasks.json',tasks)
            events=[]
            def solve(*args):
                events.append('task')
                return dict(passed=False,model_error='TimeoutError: deadline')
            def ready(*args,**kwargs):
                events.append('idle')
                return dict(status='ok')
            with patch.object(stage,'server',return_value=contextlib.nullcontext()), \
                 patch.object(repl_learning,'run_task',side_effect=solve), \
                 patch('evaluation.request',side_effect=ready):
                repl_learning.evaluate(folder,tasks,None,'repl-baseline',dict(repl_tasks=2))
            self.assertEqual(events,['task','idle','task'])

    def test_request_timeout_at_shared_deadline_is_a_failed_attempt(self):
        import repl_learning
        result=dict(error='APITimeoutError: Request timed out.', elapsed_s=240.01,
                    calls=3, output_tokens=2048, trace=[])
        with patch('polyglot_runtime.check_program',return_value=dict(passed=True)), \
             patch('recursive_agent.solve',return_value=result):
            checked=repl_learning.evaluate_one(task())
            self.assertFalse(checked['passed'])
            self.assertTrue(checked['model_error'].startswith('TimeoutError:'))
            result['elapsed_s']=1
            with self.assertRaises(RuntimeError):repl_learning.evaluate_one(task())
            result.update(error='APIConnectionError: Server stopped',elapsed_s=240.01)
            with self.assertRaises(RuntimeError):repl_learning.evaluate_one(task())

    def test_training_mix_limits_language_dominance_without_losing_repairs(self):
        from repl_learning import mix_rows
        rows=[]
        for language in stage.LANGUAGES:
            for kind,count in [('complete_code',10),('correction',2),('executed_repl_repair',4),('teacher_reference',100 if language=='python' else 0)]:
                rows.extend(dict(language=language,example_kind=kind,task_id=f'{language}-{kind}-{i}') for i in range(count))
        mixed=mix_rows(rows)
        counts={l:sum(r['language']==l for r in mixed) for l in stage.LANGUAGES}
        self.assertLessEqual(max(counts.values()),2*min(counts.values()))
        self.assertEqual(sum(r['example_kind']=='correction' for r in mixed),10)
        self.assertEqual(sum(r['example_kind']=='executed_repl_repair' for r in mixed),20)
        self.assertGreater(sum(r['example_kind']=='teacher_reference' for r in mixed),0)
        self.assertEqual(len({r['task_id'] for r in mixed}),len(mixed))

    def test_upstream_agent_limits_are_scored_without_retrying_the_worker(self):
        import repl_learning
        for error in ('ErrorThresholdExceededError: three bad cells', 'TimeoutExceededError: deadline'):
            result=dict(error=error, elapsed_s=4, calls=3, output_tokens=50, trace=[])
            with patch('polyglot_runtime.check_program',return_value=dict(passed=True)), \
                 patch('recursive_agent.solve',return_value=result):
                checked=repl_learning.evaluate_one(task())
                self.assertFalse(checked['passed'])
                self.assertIn(error,checked['model_error'])

    def test_repl_child_timeout_allows_cleanup_before_failure(self):
        import repl_learning
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            marker=Path(d)/'cleaned'
            script="import signal,time,pathlib,sys; signal.signal(signal.SIGTERM,lambda *_:(pathlib.Path(sys.argv[1]).write_text('cleaned'),sys.exit(143))); time.sleep(30)"
            with patch.object(repl_learning,'cleanup_containers') as cleanup:
                with self.assertRaises(RuntimeError):repl_learning.run_child([sys.executable,'-c',script,str(marker)],timeout=.5)
                self.assertEqual(marker.read_text(),'cleaned')
                self.assertEqual(cleanup.call_count,1)

    def test_repl_cleanup_removes_only_its_unique_owner(self):
        import repl_learning
        import subprocess
        with patch('subprocess.run',return_value=subprocess.CompletedProcess([],0,'abc\ndef\n','')) as run:
            repl_learning.cleanup_containers('a'*32)
            self.assertIn('label=local-coding-assistant.owner='+'a'*32,run.call_args_list[0].args[0])
            self.assertEqual(run.call_args_list[1].args[0],['docker','rm','-f','abc','def'])


if __name__=='__main__':unittest.main()
