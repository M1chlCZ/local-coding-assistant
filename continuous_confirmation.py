"""Resume a frozen matched confirmation comparison; never feed these tasks into training."""
import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from continuous_learning import Controller, ROOT, read
from evaluation import request
from launcher import health
from learning_session import SETTINGS, LIMITS
from recursive_agent import grade, solve
from train_adapter import atomic_json, verified_checkpoint
from training_data import registry, sha256


def compare(controller):
    controller.validate()
    state=controller.child_state();child=Path(controller.state['child'])
    reservation=state.get('confirmation',{})
    if reservation.get('consumed'):return
    if not reservation or not reservation.get('reserved_before_training'):
        raise ValueError('Fresh confirmation checks were not reserved before training')
    tasks_path=child/'confirmation-tasks.json'
    if sha256(tasks_path)!=reservation['registry_sha256']:raise ValueError('Confirmation registry digest changed')
    tasks=list(registry(tasks_path).values())
    if len(tasks)!=reservation['tasks'] or any(t['split']!='dev' for t in tasks):
        raise ValueError('Confirmation registry changed')
    baseline=Path(reservation['baseline_adapter']);current=Path(state['best_adapter'])
    baseline_hash=sha256(baseline/'adapter_model.safetensors')
    if baseline_hash!=reservation['baseline_adapter_sha256']:raise ValueError('Frozen baseline digest changed')
    for adapter in (baseline,current):verified_checkpoint(adapter,read(adapter.parent/'run.json'))
    output=controller.path/'confirmations'/child.name;output.mkdir(parents=True,exist_ok=True)
    binding={'tasks_sha256':sha256(tasks_path),'baseline_sha256':baseline_hash,
             'current_sha256':sha256(current/'adapter_model.safetensors'),'settings':SETTINGS,'limits':LIMITS}
    previous=read(output/'binding.json')
    if previous and previous!=binding:raise ValueError('Confirmation binding changed')
    atomic_json(output/'binding.json',binding)
    if read(output/'summary.json',{}).get('completed'):return
    lock=(ROOT/'.cache/learning/gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for port in (8080,8090):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Model port occupied; close chat before confirmation')
    # Reference audit is rechecked before any model call; invalid checks never get replaced post hoc.
    for task in tasks:
        reference_ok=task.get('source_reference_passed') is True if task.get('language') else grade(task,task['reference_patch'])['passed']
        if not reference_ok or grade(task,{p:task['files'][p] for p in task['editable']})['passed']:
            raise ValueError('Reserved confirmation fixture failed integrity validation')
    reports={};process=None
    try:
        for name,adapter in (('base',baseline),('baseline',baseline),('current',current)):
            if name=='current' and binding['current_sha256']==baseline_hash:
                reports[name]=reports['baseline']
                atomic_json(output/'current.json',reports[name]);continue
            report=read(output/f'{name}.json',{'tasks':[],'binding':binding})
            if report.get('binding')!=binding:raise ValueError('Confirmation report binding changed')
            ids=[t['id'] for t in tasks]
            if [t['id'] for t in report['tasks']]!=ids[:len(report['tasks'])]:
                raise ValueError('Confirmation report registry changed')
            if len(report['tasks'])<len(tasks):
                with (output/f'{name}-server.log').open('ab') as log:
                    # Same process group as comparison: Pause/Stop terminates both.
                    process=subprocess.Popen([str(ROOT/'.cache/train-env/bin/python'),
                        str(ROOT/'research/adapter_eval_server.py'),'--adapter',str(adapter)],cwd=ROOT,
                        stdout=log,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+180
                while not health(8090):
                    if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Confirmation server failed to start')
                    time.sleep(1)
                request('http://127.0.0.1:8090','/mode',{'mode':'base' if name=='base' else 'adapter'})
                for task in tasks[len(report['tasks']):]:
                    atomic_json(output/'status.json',{'status':'running','mode':name,'completed':len(report['tasks']),
                        'total':len(tasks),'task':task['id']})
                    row=solve(task,'http://127.0.0.1:8090',**SETTINGS,**LIMITS)
                    if row.get('patch'):row.update(grade(task,row['patch']))
                    report['tasks'].append(row);atomic_json(output/f'{name}.json',report)
                    print(name,task['id'],row.get('passed'),flush=True)
                process.terminate();process.wait(timeout=30);process=None
            reports[name]=report
        flags={name:{r['id']:r.get('passed') is True for r in report['tasks']} for name,report in reports.items()}
        lost=[k for k in flags['baseline'] if flags['baseline'][k] and not flags['current'][k]]
        gained=[k for k in flags['current'] if flags['current'][k] and not flags['baseline'][k]]
        atomic_json(output/'summary.json',{'completed':True,'binding':binding,'tasks':len(tasks),
            'base_passed':sum(flags['base'].values()),'baseline_passed':sum(flags['baseline'].values()),'current_passed':sum(flags['current'].values()),
            'same_weights_evaluated_once':baseline_hash==binding['current_sha256'],
            'regressions':lost,'gains':gained,'additional_gain_established':len(gained)>len(lost) and not lost,
            'scope':'Reserved public checks now consumed; never train on them. No general coding reliability claim.'})
        atomic_json(output/'status.json',{'status':'completed','completed':len(tasks),'total':len(tasks)})
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--controller',type=Path,required=True)
    compare(Controller(parser.parse_args().controller))
