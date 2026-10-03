"""Pinned public curriculum cursor. Reference code runs only in isolated Docker graders."""
import argparse
import fcntl
import json
import re
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from public_data import task_from_row
from recursive_agent import grade
from train_adapter import atomic_json
from training_data import sha256

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT/'research/opencode-source.json'


def source_manifest():
    return json.loads(MANIFEST.read_text())


def verify_source(path, shard):
    if sha256(path) != shard['sha256']:
        raise ValueError('Public source digest failed integrity verification')


class Ledger:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS seen (row_id TEXT UNIQUE, repository TEXT UNIQUE)')

    def reserve(self, task, commit=True):
        row = task.get('provenance', {}).get('row_id')
        repo = re.sub(r'-r[0-9]+$', '', task['repository'])
        cur = self.db.execute('INSERT OR IGNORE INTO seen VALUES (?,?)', (row, repo))
        if commit:
            self.db.commit()
        return cur.rowcount == 1

    def close(self):
        self.db.close()


def seed_ledger(ledger, root):
    # Reserve training AND evaluation identities, including unfinished prepared curricula.
    paths = list((root/'.cache/public-data').rglob('*tasks.json'))
    paths += list((root/'.cache/learning').glob('*/round-001/tasks.json'))
    paths += list((root/'.cache/learning').glob('*/confirmation-tasks.json'))
    paths += list((root/'.cache/learning').glob('*/development-tasks.json'))
    for path in paths:
        value = json.loads(path.read_text())
        for task in value:
            ledger.reserve(task, commit=False)
    ledger.db.commit()


def finish_batch(path, cursor, ledger):
    value = json.loads(path.read_text())
    if value['source_manifest_sha256'] != sha256(MANIFEST):
        raise ValueError('Curriculum source manifest changed')
    tasks_file = path.with_name('tasks.json')
    if sha256(tasks_file) != value['tasks_sha256']:
        raise ValueError('Prepared curriculum digest failed integrity verification')
    for task in json.loads(tasks_file.read_text()):
        ledger.reserve(task, commit=False)
    ledger.db.commit()
    atomic_json(cursor, value['next_cursor'])
    return tasks_file


def prepare_batch(path):
    """Recover one transaction or prepare the next window, without replaying used identities."""
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    cursor = path/'cursor.json'
    state = json.loads(cursor.read_text())
    manifest = source_manifest()
    if state['source_manifest_sha256'] != sha256(MANIFEST):
        raise ValueError('Public source manifest changed; review before continuing')
    batch = path/f"batch-{state['sequence']:06d}"
    ledger = Ledger(path/'seen.sqlite')
    try:
        if (batch/'manifest.json').exists():
            return finish_batch(batch/'manifest.json', cursor, ledger)
        shards = manifest['shards'];index, start = state['shard'], state['row']
        if index >= len(shards):
            raise ValueError('Approved public source exhausted; add a reviewed source before continuing')
        shard = shards[index];download = path/'source.parquet'
        if (path/'source-file.json').exists() and json.loads((path/'source-file.json').read_text()) != shard:
            download.unlink(missing_ok=True)
        if not download.exists():
            url = f"{manifest['url']}/resolve/{manifest['revision']}/{shard['file']}"
            partial = path/'source.partial'
            subprocess.run(['curl','--fail','--location','--silent','--show-error','--max-time','900',
                            '--retry','2','--output',str(partial),url],check=True)
            verify_source(partial, shard)
            partial.replace(download);atomic_json(path/'source-file.json',shard)
        verify_source(download, shard)
        from pyarrow.parquet import ParquetFile
        parquet = ParquetFile(download)
        if start >= parquet.metadata.num_rows:
            atomic_json(cursor,{**state,'shard':index+1,'row':0})
            return prepare_batch(path)
        rows=[];offset=0
        for chunk in parquet.iter_batches(batch_size=1000):
            if offset+len(chunk)>start:
                rows.extend(chunk.to_pylist()[max(0,start-offset):])
            offset+=len(chunk)
            if len(rows)>=2000:break
        rows=rows[:2000]
        candidates=[];repos=set()
        for row in rows:
            try:task=task_from_row(row)
            except ValueError:continue
            task['provenance']={**manifest,'file':shard['file'],'file_sha256':shard['sha256'],'row_id':row['id']}
            task['provenance'].pop('shards')
            if task['repository'] in repos:continue
            repos.add(task['repository']);candidates.append(task)
            if len(candidates)>=500:break
        def checked(task):
            return grade(task,task['reference_patch'])['passed'] and not grade(
                task,{p:task['files'][p] for p in task['editable']})['passed']
        tasks=[]
        # Exceptions from Docker are infrastructure failures, not disposable data failures.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for n,(task,passed) in enumerate(zip(candidates,pool.map(checked,candidates)),1):
                if passed and ledger.reserve(task,commit=False):tasks.append(task)
                print(f'Fixture checks {n}/{len(candidates)}; fresh {len(tasks)}',flush=True)
        batch.mkdir(exist_ok=True)
        atomic_json(batch/'tasks.json',tasks)
        next_cursor={**state,'row':start+len(rows),'sequence':state['sequence']+1}
        atomic_json(batch/'manifest.json',{'source_manifest_sha256':sha256(MANIFEST),
            'source':{k:v for k,v in manifest.items() if k!='shards'},'shard':shard,
            'row_start':start,'row_end':start+len(rows),'tasks_sha256':sha256(batch/'tasks.json'),
            'next_cursor':next_cursor,'scope':'Exact identity exclusions, not semantic novelty or pretraining independence.'})
        ledger.db.commit()
        atomic_json(cursor,next_cursor)
        return batch/'tasks.json'
    finally:ledger.close()


def initialize(path, root=ROOT, row=0):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    if not (path/'cursor.json').exists():
        ledger=Ledger(path/'seen.sqlite')
        try:seed_ledger(ledger,root)
        finally:ledger.close()
        atomic_json(path/'cursor.json',{'shard':0,'row':row,'sequence':1,'source_manifest_sha256':sha256(MANIFEST)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,required=True)
    args=parser.parse_args()
    with (args.pool/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('CURRICULUM='+str(prepare_batch(args.pool)),flush=True)
