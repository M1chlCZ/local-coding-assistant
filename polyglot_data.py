"""Fresh CodeContests train problems, one shared problem split across five languages."""
import argparse
import hashlib
import json
import sqlite3
import subprocess
from collections import deque
from pathlib import Path

from polyglot_runtime import LANGUAGES,FILENAMES,runner_source,check_program
from train_adapter import atomic_json
from training_data import sha256

ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/'research/codecontests-source.json'


def tasks_from_row(row):
    name,description=row.get('name'),row.get('description')
    if not isinstance(name,str) or not isinstance(description,str) or not 20<=len(description)<=4500:
        raise ValueError('Require a short English problem statement')
    if row.get('input_file') or row.get('output_file'):
        raise ValueError('Only standard input/output problems are supported')
    pairs=[];visible=[]
    for part in ('public_tests','private_tests','generated_tests'):
        suite=row.get(part,{})
        inputs,outputs=suite.get('input',[]),suite.get('output',[])
        if len(inputs)!=len(outputs):raise ValueError('Mismatched public source test pairs')
        for input_,output in zip(inputs,outputs):
            if not isinstance(input_,str) or not isinstance(output,str) or not output.strip():continue
            if len(input_)+len(output)>4000:continue
            case={'input':input_,'output':output}
            if case not in pairs:pairs.append(case)
            if part=='public_tests' and case not in visible:visible.append(case)
    if not visible or len(pairs)<3 or sum(len(c['input'])+len(c['output']) for c in pairs[:16])>20000:
        raise ValueError('Require one visible and at least three bounded source tests')
    solutions=row.get('solutions',{})
    reference=next((s for l,s in zip(solutions.get('language',[]),solutions.get('solution',[]))
                    if l==3 and isinstance(s,str) and len(s)<=10000),None)
    if not reference:raise ValueError('Require a Python 3 reference for independent source-test validation')
    identity=hashlib.sha256((name+'\n'+description).encode()).hexdigest()
    split='dev' if int(identity[:2],16)%5==0 else 'train'
    stubs={'python':"raise NotImplementedError('Implement the program')\n",
        'go':'package main\nfunc main() {}\n','typescript':"throw new Error('Implement the program');\n",
        'rust':"fn main() { panic!(\"Implement the program\"); }\n",'dart':"void main() { throw UnimplementedError(); }\n"}
    result=[]
    for language in LANGUAGES:
        filename=FILENAMES[language]
        task={'id':'codecontests-'+identity[:24]+'-'+language,'repository':'codecontests-'+identity,
            'split':split,'language':language,'filename':filename,'cases':pairs[:16],
            'prompt':f'Write a complete {language} program in {filename}. Use standard input and output, and only the standard library.\n'+description+
                '\nInspect and edit the source through the Python REPL. Run test_visible.py before and after repair. Keep test_visible.py unchanged. The REPL uses Python; the solution must use '+language+'.',
            'editable':[filename],'files':{filename:stubs[language]},
            'reference_python':reference,'provenance':{'dataset':'deepmind/code_contests','row_id':identity,'name':name}}
        task['files']['test_visible.py']=runner_source({**task,'cases':visible[:2]})+"print('VISIBLE_CHECKS_PASSED')\n"
        task['checks']=runner_source(task)
        result.append(task)
    return result


def rotate(tasks,start=0):
    """16-example rounds: eight primary examples, then balanced replay from other languages."""
    queues={l:deque(t for t in tasks if t['language']==l) for l in LANGUAGES}
    result=[];index=start
    while any(queues.values()):
        primary=LANGUAGES[index%len(LANGUAGES)];index+=1
        batch=[]
        for _ in range(8):
            if queues[primary]:batch.append(queues[primary].popleft())
        others=[l for l in LANGUAGES if l!=primary]
        for n in range(16-len(batch)):
            available=[l for l in others if queues[l]] or [l for l in LANGUAGES if queues[l]]
            if not available:break
            batch.append(queues[available[n%len(available)]].popleft())
        result.extend(batch)
    return result


def prepare_batch(path):
    from pyarrow.parquet import ParquetFile
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(MANIFEST.read_text())
    cursor=path/'cursor.json'
    state=json.loads(cursor.read_text()) if cursor.exists() else {'shard':0,'row':0,'sequence':1,'source_sha256':sha256(MANIFEST)}
    if state['source_sha256']!=sha256(MANIFEST):raise ValueError('Multilingual source manifest changed')
    folder=path/f"batch-{state['sequence']:06d}"
    if (folder/'manifest.json').exists():
        prepared=json.loads((folder/'manifest.json').read_text())
        if sha256(folder/'tasks.json')!=prepared['tasks_sha256']:raise ValueError('Prepared multilingual tasks digest changed')
        db=sqlite3.connect(path/'seen.sqlite');db.execute('CREATE TABLE IF NOT EXISTS seen (problem TEXT PRIMARY KEY)')
        try:
            for task in json.loads((folder/'tasks.json').read_text()):db.execute('INSERT OR IGNORE INTO seen VALUES (?)',(task['repository'],))
            db.commit()
        finally:db.close()
        atomic_json(cursor,prepared['next_cursor']);return folder/'tasks.json'
    shards=manifest['shards']
    if state['shard']>=len(shards):raise ValueError('Approved multilingual source exhausted')
    shard=shards[state['shard']];source=path/'source.parquet'
    if (path/'source-file.json').exists() and json.loads((path/'source-file.json').read_text())!=shard:source.unlink(missing_ok=True)
    if not source.exists():
        partial=path/'source.partial'
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','2','--max-time','900',
            '--output',str(partial),f"{manifest['url']}/resolve/{manifest['revision']}/{shard['file']}"],check=True)
        if sha256(partial)!=shard['sha256']:raise ValueError('Multilingual source digest failed integrity verification')
        partial.replace(source);atomic_json(path/'source-file.json',shard)
    if sha256(source)!=shard['sha256']:raise ValueError('Multilingual source digest failed integrity verification')
    parquet=ParquetFile(source)
    if state['row']>=parquet.metadata.num_rows:
        atomic_json(cursor,{**state,'shard':state['shard']+1,'row':0});return prepare_batch(path)
    rows=[];offset=0
    for batch in parquet.iter_batches(batch_size=100):
        if offset+len(batch)>state['row']:rows.extend(batch.to_pylist()[max(0,state['row']-offset):])
        offset+=len(batch)
        if len(rows)>=100:break
    rows=rows[:100];tasks=[]
    db=sqlite3.connect(path/'seen.sqlite');db.execute('CREATE TABLE IF NOT EXISTS seen (problem TEXT PRIMARY KEY)')
    try:
        for row in rows:
            try:versions=tasks_from_row(row)
            except ValueError:continue
            identity=versions[0]['repository']
            if db.execute('SELECT 1 FROM seen WHERE problem=?',(identity,)).fetchone():continue
            reference=versions[0]
            checked=check_program('python',reference['reference_python'],reference['cases'])
            if not checked['passed']:continue
            for task in versions:
                task['source_reference_passed']=True
                task['source_reference_sha256']=hashlib.sha256(task.pop('reference_python').encode()).hexdigest()
                task['provenance'].update({k:manifest[k] for k in ('revision','url','license','attribution')})
                task['provenance'].update(file=shard['file'],file_sha256=shard['sha256'])
            tasks.extend(versions);db.execute('INSERT INTO seen VALUES (?)',(identity,))
            print(f'Validated fresh multilingual problems {len(tasks)//5}',flush=True)
        train=rotate([t for t in tasks if t['split']=='train'],start=(state['sequence']-1)*5)
        development=[t for t in tasks if t['split']=='dev']
        atomic_json(folder/'tasks.json',train+development)
        next_={**state,'row':state['row']+len(rows),'sequence':state['sequence']+1}
        atomic_json(folder/'manifest.json',{'tasks_sha256':sha256(folder/'tasks.json'),'next_cursor':next_,
            'source_manifest_sha256':sha256(MANIFEST),'languages':list(LANGUAGES),
            'scope':'Shared problem-level splits; exact identity exclusions only. Public-source and pretraining overlap is unknown.'})
        db.commit();atomic_json(cursor,next_);return folder/'tasks.json'
    finally:db.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pool',type=Path,required=True)
    args=p.parse_args();print('CURRICULUM='+str(prepare_batch(args.pool)),flush=True)
