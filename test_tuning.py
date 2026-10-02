"""Focused tuning check: python3 test_tuning.py; Docker fixtures checked on the PC."""
import json
import math
from pathlib import Path
import learning_session as session
import train_adapter as trainer

curriculum=Path(__file__).parent/'research/repair_tasks.json'
assert curriculum.is_file(), 'Broader repair curriculum is missing'
tasks=json.loads(curriculum.read_text())
assert len([t for t in tasks if t['split']=='train'])==20
assert len([t for t in tasks if t['split']=='dev'])==10
assert len({t['id'] for t in tasks})==30
assert all('test_visible.py' in t['files'] and 'checks' in t for t in tasks)
for t in tasks:
    for source in [*t['files'].values(), *t['reference_patch'].values(), t['checks']]:
        compile(source,'<fixture>','exec')
assert len(session.round_tasks(1))==30
assert all(t['split']=='train' and 'test_visible.py' in t['files'] for t in session.round_tasks(1))
assert len(session.development_tasks())==15
assert trainer.training_steps(18,80,1)==math.ceil(18/4)
assert trainer.training_steps(200,80,1)==50
assert trainer.training_steps(1000,80,1)==80
args=trainer.parser().parse_args(['--dataset','data','--model','test','--revision','a'*40,'--output','out'])
assert args.max_epochs==1 and args.learning_rate==5e-5
assert session.should_stop_for_quality([{'accepted':False}]*3)
assert not session.should_stop_for_quality([{'accepted':False}]*2)
assert not session.should_stop_for_quality([{'accepted':False},{'accepted':True},{'accepted':False}])
# A partial or duplicate report cannot qualify for the expanded evaluation.
def report(n,passed):
    return {'tasks':[{'id':str(i),'passed':i in passed} for i in range(n)]}
assert session.accepted(report(15,{0}),report(15,set()),expected=15)
assert not session.accepted(report(14,{0}),report(14,set()),expected=15)
for invalid in [('max_epochs',0),('max_epochs',4),('learning_rate',float('nan')),('learning_rate',1)]:
    original=getattr(args,invalid[0]); setattr(args,*invalid)
    try: trainer.validate_args(args)
    except ValueError: pass
    else: raise AssertionError('Invalid tuning setting accepted')
    setattr(args,invalid[0],original)
print('PASS: broader curriculum, short candidate training, and quality plateau stop')

# Execute the exact derived REPL code in a temporary workspace using the local interpreter.
# Docker isolation is exercised separately on the PC; this checks the training behavior.
import contextlib
import io
import tempfile
import traceback
from types import SimpleNamespace
from unittest.mock import patch
import training_data as data
from test_rlm_tasks import run
class Workspace:
    def __init__(self, context_payload, **kwargs):
        self.temporary=tempfile.TemporaryDirectory(); self.root=Path(self.temporary.name)
        self.state={'context':context_payload,'answer':{'content':'','ready':False}}
        for name,content in context_payload['files'].items(): (self.root/name).write_text(content)
    def execute_code(self,code):
        code=code.replace('/workspace',str(self.root))
        stdout=io.StringIO(); stderr=io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                exec(code,self.state,self.state)
        except BaseException: traceback.print_exc(file=stderr)
        return SimpleNamespace(stdout=stdout.getvalue(),stderr=stderr.getvalue())
    def cleanup(self): self.temporary.cleanup()
def local_grade(task, changes):
    result=run({**task['files'],**changes},task['checks'])
    return {'passed':result.returncode==0,'feedback':result.stdout+result.stderr}
task=next(t for t in tasks if t['id']=='tuning-escaped-delimiter')
trace=[{'kind':'root','messages':[{'role':'system','content':'Use REPL'}, {'role':'user','content':'Turn 1/8:'}]}]
with patch('recursive_agent.grade',local_grade),patch('recursive_agent.Sandbox',Workspace):
    calls=data.repair_calls(task,trace,task['reference_patch'])
    assert len(calls)==4, 'Repair examples must demonstrate failure, patching and successful retest'
    assert 'AssertionError' in str(calls[2][0])
    assert 'VISIBLE_CHECKS_PASSED' in str(calls[3][0])
    assert 'reference_patch' not in str(calls)
    assert "answer['ready'] = True" in calls[-1][1]
    for _,response in calls: compile(response.split('```repl\n',1)[1].rsplit('```',1)[0],'<target>','exec')
print('PASS: four-turn repair targets execute quoted code, reproduce a failure and retest successfully')

# A paused teacher comparison resumes at the first unfinished development task.
assert hasattr(session.Session,'teacher_baseline'), 'Compatible teacher comparison is missing'
with tempfile.TemporaryDirectory() as temporary:
    folder=Path(temporary)
    dev=session.development_tasks()
    trainer.atomic_json(folder/'teacher-dev.json',{'tasks':[{'id':dev[0]['id'],'passed':False}]})
    worker=object.__new__(session.Session); worker.state={}; worker.save=lambda **kwargs:None
    worker.interrupted=lambda:False
    observed=[]
    def solve(task,*args,**kwargs):
        observed.append(task['id']);return {'id':task['id'],'passed':False}
    with patch('recursive_agent.solve',solve):
        assert worker.teacher_baseline(folder)
    assert observed==[t['id'] for t in dev[1:]]
    assert len(json.loads((folder/'teacher-dev.json').read_text())['tasks'])==15
print('PASS: teacher comparison resumes without repeating completed tasks')

# Preserve the teacher's actual successful actions instead of replacing them with a script.
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary); folder=root/'round-001'; folder.mkdir()
    task=next(t for t in tasks if t['id']=='tuning-escaped-delimiter')
    trainer.atomic_json(folder/'tasks.json',[task])
    response='```repl\npatch = '+repr(task['reference_patch'])+"\nanswer['content'] = patch\nanswer['ready'] = True\n```"
    trace={'kind':'root','messages':[{'role':'system','content':'Use REPL'},
           {'role':'user','content':'Turn 1/8:'}],'response':response,'input_tokens':10,'output_tokens':10}
    trainer.atomic_json(folder/'teacher.json',{'schema_version':1,'split':'train',
        'tasks_sha256':data.sha256(folder/'tasks.json'),'tasks':[dict(task,passed=True,mode='rlm',trace=[trace],patch=task['reference_patch'])]})
    worker=object.__new__(session.Session); worker.state={'round':1}; worker.path=root
    worker.save=lambda **kwargs:None
    with patch('recursive_agent.grade',local_grade),patch('recursive_agent.Sandbox',Workspace):
        worker.dataset(folder)
    rows=data.load_verified(folder/'training.jsonl',folder/'tasks.json')
    assert [r['messages'][-1]['content'] for r in rows]==[response], 'Actual successful teacher actions were replaced'
print('PASS: learning data preserves the actual verified teacher trajectory')
