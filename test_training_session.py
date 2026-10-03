"""CPU self-check. Optional CUDA recovery check: python test_training_session.py DATA TASKS [WARM_ADAPTER]."""
import json
import tempfile
import os
import subprocess
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import train_adapter as adapter

assert hasattr(adapter, 'latest_checkpoint'), 'Complete checkpoint recovery is missing'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: adapter.atomic_json(root/'control.json', {'request':i}), range(20)))
    assert json.loads((root/'control.json').read_text())['request'] in range(20)
    assert not list(root.glob('*.tmp'))
    binding={'model':'Qwen/test','dataset_sha256':'1'*64,'max_steps':10}
    first=root/'checkpoint-2'
    first.mkdir()
    for name in adapter.CHECKPOINT_FILES:
        (first/name).write_text(json.dumps({'global_step':2}) if name=='trainer_state.json' else name)
    adapter.complete_checkpoint(first,binding)
    assert adapter.latest_checkpoint(root,binding)==first
    unfinished=root/'checkpoint-3'
    unfinished.mkdir()
    (unfinished/'adapter_model.safetensors').write_text('partial')
    assert adapter.latest_checkpoint(root,binding)==first
    try: adapter.latest_checkpoint(root,{**binding,'dataset_sha256':'2'*64})
    except ValueError: pass
    else: raise AssertionError('Changed dataset accepted')
    (first/'optimizer.pt').write_text('changed')
    try: adapter.latest_checkpoint(root,binding)
    except ValueError: pass
    else: raise AssertionError('Changed optimizer accepted')
    args=adapter.parser().parse_args(['--dataset',str(root/'data.jsonl'),'--model','Qwen/test',
        '--revision','1'*40,'--output',str(root),'--resume','--save-steps','2'])
    adapter.validate_args(args)
from unittest.mock import patch
import learning_session as session
import training_data as data

variants = session.round_tasks(3)
assert len(variants) == 30 and all(t['split'] == 'train' for t in variants)
assert variants == session.round_tasks(3)
query=next(t for t in variants if t['id'].startswith('query-updates'))
assert 'parts.query,' in query['reference_patch']['links.py']
assert 'from query import' in query['files']['links.py']
for task in variants:
    for source in task['files'].values(): compile(source, '<source>', 'exec')
    for source in task['reference_patch'].values(): compile(source, '<reference>', 'exec')
    compile(task['checks'], '<checks>', 'exec')

def report(passed):
    return {'tasks': [{'id': str(i), 'passed': i in passed} for i in range(5)]}
assert session.accepted(report({0}), report(set()))
assert not session.accepted(report({1}), report({0}))
assert not session.accepted(report({0}), report(set()), report({0}))
assert not session.accepted({'tasks': report({0})['tasks'][:4]}, report(set()))
assert session.accepted(report({0,1}), report({0}), report({0}))
# CPU session checks use a temporary repository, never the live GPU lock.
def isolated_repository(root):
    repository=root/'test-repository'
    for name in session.SOURCE_FILES+('research/rlm_tasks.json',):
        destination=repository/name; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(session.ROOT/name,destination)
    return patch('learning_session.ROOT',repository)

# Lock ownership, durable controls, and phase recovery require no GPU.
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    with isolated_repository(root):
        worker=session.Session(root,12,Path('/teacher.gguf'))
        worker.save(phase='train')
        adapter.atomic_json(root/'command.json', {'action':'pause'})
        assert worker.interrupted()
        try: session.Session(root,12,Path('/teacher.gguf'))
        except BlockingIOError: pass
        else: raise AssertionError('Two workers acquired the same session')
        worker.lock.close()
        worker.gpu_lock.close()
        resumed=session.Session(root,1,Path('/other.gguf'))
        assert resumed.state['phase']=='train' and resumed.state['limit_seconds']==43200
        assert resumed.state['teacher_model']=='/teacher.gguf'
        resumed.save(status='paused')
        before=resumed.state['active_seconds']
        resumed.save()
        assert resumed.state['active_seconds']==before
        resumed.lock.close()
        resumed.gpu_lock.close()

# A shutdown between dataset and manifest writes preserves the partial file and regenerates it.
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    with isolated_repository(root):
        worker=session.Session(root,12,Path('/teacher.gguf'))
        folder=root/'round-001';folder.mkdir()
        adapter.atomic_json(folder/'teacher.json', {'tasks':[{'passed':True}]})
        adapter.atomic_json(folder/'tasks.json', [])
        (folder/'training.jsonl').write_text('partial')
        def fake_export(reports,tasks,output,repairs):
            assert not repairs and not output.exists()
            output.write_text('complete')
        with patch('learning_session.export',side_effect=fake_export), patch('learning_session.load_verified'):
            worker.dataset(folder)
        assert worker.state['phase']=='train'
        assert len(list(folder.glob('training-incomplete-*.jsonl')))==1
        assert (folder/'training.jsonl').read_text()=='complete'
        worker.lock.close();worker.gpu_lock.close()
print('PASS: checkpoint integrity, split separation, verified repair targets, acceptance, exclusive worker, and recovery')

with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    extra={**session.development_tasks()[0],'id':'public-extra-dev'}
    adapter.atomic_json(root/'development-tasks.json',[extra])
    with isolated_repository(root):
        worker=session.Session(root,12,Path('/teacher.gguf'))
        assert worker.state.get('development_tasks_sha256')==data.sha256(root/'development-tasks.json'), 'Extra checks are not bound to the session'
        worker.save()
        worker.lock.close();worker.gpu_lock.close()
        (root/'development-tasks.json').write_text('[]')
        try: session.Session(root,12,Path('/teacher.gguf'))
        except ValueError: pass
        else: raise AssertionError('Changed quality checks accepted on resume')
print('PASS: extra development tasks are immutable across resume')

with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    with isolated_repository(root):
        worker=session.Session(root,12,Path('/teacher.gguf'),research=True,fast_reject=True)
        worker.save(status='paused')
        worker.lock.close();worker.gpu_lock.close()
        resumed=session.Session(root,1,Path('/other.gguf'))
        assert resumed.state['fast_reject']
        assert resumed.state['research'] is True and resumed.state['limit_seconds']==43200
        resumed.lock.close();resumed.gpu_lock.close()
print('PASS: research mode and its time limit survive a worker restart')

if len(sys.argv)>1:
    if len(sys.argv) not in (3,4):
        raise SystemExit('Use DATA TASKS [WARM_ADAPTER], with the cached model and pinned training environment')
    with tempfile.TemporaryDirectory(prefix='lca-checkpoint-') as directory:
        root=Path(directory); pause=root/'pause'; pause.touch()
        output=root/'adapter'
        command=[sys.executable,str(session.ROOT/'train_adapter.py'),'--dataset',sys.argv[1],
            '--tasks',sys.argv[2],'--model',session.MODEL,'--revision',session.REVISION,
            '--output',str(output),'--max-steps','2','--max-epochs','3','--max-length','4096','--save-steps','1',
            '--pause-file',str(pause)]
        if len(sys.argv)==4: command+=['--warm-start',sys.argv[3]]
        environment={**os.environ,'HF_HUB_OFFLINE':'1'}
        subprocess.run(command,check=True,env=environment)
        assert json.loads((output/'training.json').read_text())['global_step']==1
        pause.unlink()
        subprocess.run(command+['--resume'],check=True,env=environment)
        result=json.loads((output/'training.json').read_text())
        assert result['global_step']==2 and result['resumed_from'].endswith('checkpoint-1')
        digest=data.sha256(output/'adapter_model.safetensors')
        (output/'training.json').unlink()  # Simulate shutdown after the last checkpoint, before final metadata.
        subprocess.run(command+['--resume'],check=True,env=environment)
        result=json.loads((output/'training.json').read_text())
        assert result['global_step']==2 and result['recovered_completed_checkpoint']
        assert data.sha256(output/'adapter_model.safetensors')==digest
    print('PASS: CUDA pause, optimizer resume, and final-checkpoint recovery without an extra step')
