"""Compare the pinned student and a local GGUF on the expanded development set."""
import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from learning_session import ROOT, LIMITS, SETTINGS, development_tasks
from launcher import health
from recursive_agent import grade, solve
from train_adapter import atomic_json
from training_data import sha256
from evaluation import request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter',required=True,type=Path)
    parser.add_argument('--gguf',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    lock=(ROOT/'.cache/learning/gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources={str(p.relative_to(ROOT)):sha256(p) for p in
             (ROOT/'research/rlm_tasks.json',ROOT/'research/repair_tasks.json',ROOT/'recursive_agent.py',ROOT/'evaluation.py',ROOT/'rlm_worker.py')}
    summary={'scope':'15 authored development repairs, shared RLM prompt and budgets. Original holdout unused. Small pilot, not general coding reliability.',
             'sources':sources,'limits':LIMITS,'settings':SETTINGS,'models':{},
             'student':'Qwen/Qwen3-4B NF4',
             'student_revision':'1cfa9a7208912126459214e8b04321603b3df60c',
             'old_adapter_sha256':sha256(args.adapter/'adapter_model.safetensors'),
             'gguf_name':args.gguf.name,'gguf_sha256':sha256(args.gguf)}
    tasks=development_tasks()
    for server,modes,port in [
        ([str(ROOT/'.cache/train-env/bin/python'),'research/adapter_eval_server.py','--adapter',str(args.adapter)],('base','old-adapter'),8090),
        ([str(ROOT/'.cache/rlm-env/bin/python'),'wsl_server.py','--model',str(args.gguf),'--cpu-ffn','0'],('mimo-9b',),8080)]:
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',port))==0: raise ValueError(f'Port {port} is occupied')
        with (args.output/f'server-{port}.log').open('wb') as log:
            child=subprocess.Popen(server,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+180
            while not health(port):
                if child.poll() is not None or time.monotonic()>deadline: raise RuntimeError('Comparison server failed to start')
                time.sleep(1)
            for mode in modes:
                if port==8090: request(f'http://127.0.0.1:{port}','/mode',{'mode':'adapter' if mode=='old-adapter' else 'base'})
                report={'tasks':[],'split':'dev','sources':sources,'settings':SETTINGS,'limits':LIMITS}
                for task in tasks:
                    row=solve(task,f'http://127.0.0.1:{port}',**SETTINGS,**LIMITS)
                    if row.get('patch'): row.update(grade(task,row['patch']))
                    report['tasks'].append(row)
                    atomic_json(args.output/f'{mode}.json',report)
                    print(mode,task['id'],'PASS' if row['passed'] else 'FAIL',flush=True)
                rows=report['tasks']
                summary['models'][mode]={'passed':sum(r['passed'] for r in rows),'total':len(rows),
                    'seconds':round(sum(r['elapsed_s'] for r in rows),2),'calls':sum(r['calls'] for r in rows),
                    'output_tokens':sum(r['output_tokens'] for r in rows),
                    'subcalls':sum(c.get('kind')=='subcall' for r in rows for c in r['trace'])}
                atomic_json(args.output/'summary.json',summary)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
    print(json.dumps(summary['models']),flush=True)


if __name__=='__main__': main()
