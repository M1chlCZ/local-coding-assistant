"""Matched, resumable direct HumanEval/MultiPL-E audit, kept outside training."""
import argparse
import fcntl
import json
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import chat,chat_batch,check_code,request
from launcher import health
from polyglot_runtime import LANGUAGES,check_program,image
from research.student_benchmark import MODEL,REVISION,load_tasks as python_tasks
from train_adapter import atomic_json,verified_checkpoint
from training_data import sha256

SOURCES=('research/polyglot_benchmark.py','polyglot_runtime.py','research/multipl-e-source.json',
         'research/adapter_eval_server.py','evaluation.py','launcher.py','train_adapter.py','training_data.py')


def load_tasks():
    from pyarrow.parquet import read_table
    metadata=json.loads((ROOT/'research/multipl-e-source.json').read_text())
    folder=ROOT/'.cache/benchmarks/multipl-e-source';folder.mkdir(parents=True,exist_ok=True)
    original_metadata,original=python_tasks()
    tasks={'python':[{'id':t['task_id'],'prompt':t['prompt'],'tests':t['test']+'\ncheck('+t['entry_point']+')',
                     'entry_point':t['entry_point']} for t in original]}
    for language,info in metadata['files'].items():
        path=folder/(language+'.parquet')
        if not path.exists():
            partial=path.with_suffix('.partial')
            subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','2','--max-time','120',
                '--output',str(partial),f"{metadata['url']}/resolve/{metadata['revision']}/{info['file']}"],check=True)
            if sha256(partial)!=info['sha256']:raise ValueError('Multilingual benchmark source digest failed')
            partial.replace(path)
        if sha256(path)!=info['sha256']:raise ValueError('Multilingual benchmark source digest failed')
        rows=read_table(path).to_pylist()
        tasks[language]=[{'id':t['name'],'prompt':t['prompt'],'tests':t['tests'],
                         'entry_point':t['name'].split('_',2)[2]} for t in rows]
        if len({t['id'] for t in tasks[language]})!=len(rows) or len(rows)<100:
            raise ValueError('Full multilingual benchmark registry is missing')
    return {'python':original_metadata,'multipl_e':metadata},tasks


def program(language,task,content):
    match=re.search(r'```[^\n]*\n(.*?)```',content,re.S)
    code=(match.group(1) if match else content).strip()
    if language=='python':
        from benchmark import assemble_solution
        return assemble_solution(task,content)
    # Request a complete source/function; upstream Rust and Dart tests close a completion stub.
    tests=task['tests']
    if language in ('rust','dart') and tests.lstrip().startswith('}'):
        tests=tests.lstrip()[1:]
    if language=='go':
        imports=re.search(r'(?s)\A(.*?import\s*\(.*?\)\n)',task['prompt'])
        if not re.search(r'(?m)^package\s+',code):
            code=(imports.group(1) if imports else 'package main\nimport("testing";"fmt")\n')+code
        declarations='\n'.join(re.findall(r'(?m)^\s*import\s*(?:\([^)]*\)|[^\n]*)',code))
        # Restore libraries used by the appended tests, never missing algorithm imports.
        missing=[name for name in ('testing','fmt') if re.search(r'\b'+name+r'\.',tests)
                 and '"'+name+'"' not in declarations]
        if missing:
            code=re.sub(r'(?m)^(package\s+\w+[^\n]*\n)',
                lambda m:m[0]+'import ('+';'.join(json.dumps(n) for n in missing)+')\n',code,count=1)
    return code+'\n'+tests


def checked(language,task,content):
    source=program(language,task,content)
    if language=='python':return check_code(source,task['tests'])
    filename={'go':'solution_test.go','typescript':'benchmark.ts','rust':'benchmark.rs','dart':'benchmark.dart'}[language]
    return check_program(language,source,filename=filename)


def summarize(binding,tasks,reports):
    result={};macros={mode:[] for mode in ('base','adapter')}
    for language in binding['languages']:
        expected=[t['id'] for t in tasks[language]];flags={};models={}
        for mode in ('base','adapter'):
            rows=reports[mode][language]
            if [r['id'] for r in rows]!=expected:raise ValueError('Both models must finish every task in every language')
            flags[mode]={r['id']:r.get('passed') is True for r in rows}
            passed=sum(flags[mode].values());score=passed/len(rows)
            models[mode]={'passed':passed,'total':len(rows),'pass_at_1':score,
                'median_answer_seconds':statistics.median(r['elapsed_s'] for r in rows)}
            if binding.get('generation_batch_size',1)>1:
                batches={r['generation_batch_id']:r['generation_batch_seconds'] for r in rows}
                seconds=sum(batches.values());tokens=sum(r['output_tokens'] for r in rows)
                models[mode].update(generation_wall_seconds=round(seconds,3),output_tokens=tokens,
                    output_tokens_per_generation_second=round(tokens/seconds,3) if seconds else 0)
            macros[mode].append(score)
        result[language]={'models':models,'gained':[i for i in expected if flags['adapter'][i] and not flags['base'][i]],
            'lost':[i for i in expected if flags['base'][i] and not flags['adapter'][i]]}
    return {'completed':True,'binding':binding,'languages':result,
        'macro_pass_at_1':{m:sum(v)/len(v) for m,v in macros.items()},
        'scope':'Full original HumanEval (Python) and pinned MultiPL-E HumanEval translations. One greedy attempt; no tools or repairs. Language totals differ.',
        'purpose':'Reporting only. Never train on these tasks or use scores for checkpoint selection.',
        'limitations':['Benchmark/public-source/pretraining overlap is unknown.','Function tasks do not measure full repository or framework work.',
                       'Prompt, toolchain and generation settings differ from published leaderboards; compare only matched runs.']}


def run(adapter,output,limit=None,batch_size=1):
    if batch_size not in (1,2,4,8,16):raise ValueError('Use a checked batch size of 1, 2, 4, 8 or 16')
    metadata,tasks=load_tasks()
    if limit:
        tasks={l:ts[:limit] for l,ts in tasks.items()}
    adapter=adapter.resolve();verified_checkpoint(adapter,json.loads((adapter.parent/'run.json').read_text()))
    binding={'model':MODEL,'revision':REVISION,'quantization':'NF4','languages':list(LANGUAGES),
        'adapter_sha256':sha256(adapter/'adapter_model.safetensors'),'adapter_config_sha256':sha256(adapter/'adapter_config.json'),
        'datasets':metadata,'container_image':image(),'max_output_tokens':1024,'temperature':0,'seed':42,'thinking':False,
        'attempts_per_task':1,'smoke_limit':limit,'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'prompt':'Return complete source for the supplied function and needed imports/helpers. No tests, main, examples or reasoning.'}
    if batch_size>1:binding['generation_batch_size']=batch_size
    output.mkdir(parents=True,exist_ok=True)
    prior=json.loads((output/'binding.json').read_text()) if (output/'binding.json').exists() else binding
    if prior!=binding:raise ValueError('Multilingual audit binding changed; use a new output folder')
    atomic_json(output/'binding.json',binding)
    reports={m:{} for m in ('base','adapter')}
    for mode in reports:
        for l in LANGUAGES:
            p=output/(mode+'-'+l+'.json')
            reports[mode][l]=json.loads(p.read_text()) if p.exists() else []
            if [r['id'] for r in reports[mode][l]]!=[t['id'] for t in tasks[l][:len(reports[mode][l])]]:
                raise ValueError('Multilingual audit progress does not match fixed task order')
    if (output/'summary.json').exists():
        expected=summarize(binding,tasks,reports)
        if json.loads((output/'summary.json').read_text())!=expected:raise ValueError('Completed multilingual audit digest changed')
        print('Complete multilingual audit reused; no generation repeated',flush=True);return
    lock=(ROOT/'.cache/learning/gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    server=None
    try:
        for port in (8080,8090):
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('GPU model port is occupied')
        with (output/'server.log').open('ab') as log:
            server=subprocess.Popen([str(ROOT/'.cache/train-env/bin/python'),str(ROOT/'research/adapter_eval_server.py'),
                '--adapter',str(adapter)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+180
        while not health(8090):
            if server.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Multilingual audit student server failed to start')
            time.sleep(1)
        for mode in ('base','adapter'):
            request('http://127.0.0.1:8090','/mode',{'mode':mode})
            for l in LANGUAGES:
                rows=reports[mode][l]
                for offset in range(len(rows),len(tasks[l]),batch_size):
                    chunk=tasks[l][offset:offset+batch_size]
                    atomic_json(output/'status.json',{'status':'running','phase':mode,'language':l,'completed':len(rows),'total':len(tasks[l]),'smoke':bool(limit)})
                    responses,elapsed=chat_batch('http://127.0.0.1:8090',[[
                        {'role':'system','content':'You are an English coding assistant. '+binding['prompt']+' Use '+l+'.'},
                        {'role':'user','content':task['prompt']}] for task in chunk],max_tokens=1024)
                    contents=[response['choices'][0]['message'].get('content') or '' for response in responses]
                    with ThreadPoolExecutor(max_workers=min(2,len(chunk))) as pool:
                        results=list(pool.map(lambda pair:checked(l,*pair),zip(chunk,contents)))
                    batch_id=uuid.uuid4().hex
                    for task,response,content,result in zip(chunk,responses,contents,results):
                        if result.get('exit_code') in (125,126,127):raise RuntimeError('Benchmark container infrastructure failed')
                        row={'id':task['id'],'content':content,'elapsed_s':elapsed,'output_tokens':response['usage']['completion_tokens'],**result}
                        if batch_size>1:row.update(generation_batch_id=batch_id,generation_batch_size=len(chunk),generation_batch_seconds=elapsed)
                        rows.append(row)
                        print(mode,l,task['id'],'PASS' if result['passed'] else 'FAIL',flush=True)
                    atomic_json(output/(mode+'-'+l+'.json'),rows)
        summary=summarize(binding,tasks,reports);summary['completed']=not bool(limit)
        if limit:summary['scope']='Multilingual harness smoke subset; not a complete standardized benchmark score'
        atomic_json(output/('smoke-summary.json' if limit else 'summary.json'),summary)
        atomic_json(output/'status.json',{'status':'completed','smoke':bool(limit),'languages':list(LANGUAGES)})
    finally:
        if server and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait()
        lock.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--smoke-limit',type=int,choices=range(1,11))
    parser.add_argument('--batch-size',type=int,choices=(1,2,4,8,16),default=1)
    args=parser.parse_args();signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    run(args.adapter,args.output,args.smoke_limit,args.batch_size)
