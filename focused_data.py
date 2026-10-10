"""Prepare pinned, publisher-tested Go/TypeScript repairs; never execute source metadata."""
import argparse
from collections import Counter, defaultdict, deque
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parent
DATASET = 'nebius/SWE-rebench-V2'
REVISION = '10483de0f50fe5da545942705a76c6150171af7f'
SOURCE_FILE = 'data/train-00000-of-00001.parquet'
SOURCE_SHA256 = '0e0bf9355f892ad74ae98d4e1c404f39fd6654a8e351ee3e6ab162e4a64cd3ad'
SOURCE_BYTES = 428839266
LICENSES = {'MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause', 'ISC', 'MIT-0', 'Unlicense'}
BANNED = ('humaneval', 'human-eval', 'multipl-e', 'multi-pl-e', 'mbpp', 'bigcode-evaluation')
SYSTEM = ('You are an English coding assistant. Fix the reported problem using the supplied source excerpts. '
          'Return only a unified diff with the original file paths and line numbers. Do not change tests. '
          'Excerpts are partial files; keep surrounding code unchanged.')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): result.update(chunk)
    return result.hexdigest()


def atomic(path, content):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name+'.tmp')
    partial.write_text(content, encoding='utf-8'); partial.replace(path)


def family(repo):
    # Forks with the same repository name share a split, regardless of owner/case.
    return repo.lower().removesuffix('.git').split('/')[-1]


def split_for(repo):
    value = int(digest('focused-v1:'+family(repo))[:8], 16) % 100
    return 'test' if value < 10 else 'dev' if value < 20 else 'train'


def shingles(text):
    words = re.findall(r'[a-z0-9_]+', text.lower())
    return {tuple(words[i:i+13]) for i in range(max(0,len(words)-12))}


def safe_path(name, language):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or not name:
        raise ValueError('unsafe_path')
    expected = (('.go',) if language == 'go' else ('.ts', '.tsx')) + ('.md', '.json', '.yaml', '.yml', '.mod', '.sum')
    if not name.endswith(expected): raise ValueError('other_file_language')
    if any(part.lower() in ('test', 'tests', '__tests__', 'fixtures', '__fixtures__', 'testdata') for part in path.parts):
        raise ValueError('test_patch')
    if re.search(r'(_test\.go|[.-](test|spec)\.tsx?)$', name): raise ValueError('test_patch')


def excerpts(patch, language):
    """Recover only pre-fix hunk content, preserving positions and never exposing added lines."""
    if len(patch) > 10000 or any(x in patch for x in ('Binary files', 'GIT binary patch', 'rename from ', 'deleted file mode')):
        raise ValueError('unsupported_patch')
    lines = patch.splitlines(); result=[]; paths=[]; hunk_count=0; old_left=new_left=0; active=False
    for line in lines:
        if line.startswith('diff --git '):
            if old_left or new_left: raise ValueError('malformed_hunk')
            match=re.fullmatch(r'diff --git a/(\S+) b/(\S+)',line)
            if not match or match[1]!=match[2]:raise ValueError('unsupported_path_change')
            safe_path(match[1],language);paths.append(match[1]);result.append('\nFile: '+match[1]);active=False
        elif line.startswith('@@ '):
            if old_left or new_left:raise ValueError('malformed_hunk')
            match=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)',line)
            if not match or not paths:raise ValueError('malformed_hunk')
            old_left=int(match[2] or 1);new_left=int(match[4] or 1);hunk_count+=1;active=True
            if old_left==0:result.append('[No previous lines here; insert new source]')
            result.append('Original lines '+match[1]+', '+str(old_left)+':'+match[5])
        elif active:
            if line.startswith('\\ No newline'):continue
            if line.startswith(' '):old_left-=1;new_left-=1;result.append(line[1:])
            elif line.startswith('-'):old_left-=1;result.append(line[1:])
            elif line.startswith('+'):new_left-=1
            else:raise ValueError('malformed_hunk')
            if min(old_left,new_left)<0:raise ValueError('malformed_hunk')
    if not any(name.endswith(('.go',) if language=='go' else ('.ts','.tsx')) for name in paths):
        raise ValueError('no_primary_language_source')
    if old_left or new_left or not hunk_count or len(paths)>4 or hunk_count>10:
        raise ValueError('patch_scope')
    return '\n'.join(result), paths


def convert(row, exclusions):
    if row.get('language') not in ('go','ts'):raise ValueError('language')
    language={'go':'go','ts':'typescript'}[row['language']]
    if row.get('license') not in LICENSES:raise ValueError('license')
    meta=(row.get('meta') or {}).get('llm_metadata') or {}
    if meta.get('code')!='A' or any((meta.get('detected_issues') or {}).values()):raise ValueError('publisher_quality')
    if not isinstance(row.get('FAIL_TO_PASS'),list) or not row['FAIL_TO_PASS'] or not row.get('test_patch'):
        raise ValueError('execution_evidence')
    repo=row.get('repo','');statement=row.get('problem_statement','');patch=row.get('patch','')
    if not re.fullmatch(r'[\w.-]+/[\w.-]+',repo) or not re.fullmatch(r'[0-9a-f]{40}',row.get('base_commit','')):
        raise ValueError('source_identity')
    if not isinstance(statement,str) or not 20<=len(statement)<=8000:raise ValueError('statement_size')
    if any(marker in (repo+' '+statement+' '+patch).lower() for marker in BANNED):raise ValueError('benchmark_source')
    if exclusions & shingles(statement+'\n'+patch):raise ValueError('benchmark_overlap')
    original,paths=excerpts(patch,language)
    prompt=('Repair this '+language+' code. The excerpts below show the original code before the fix.\n\n'
            'Reported problem:\n'+statement+'\n\nOriginal source excerpts:\n'+original)
    if len(prompt)+len(patch)>16000:raise ValueError('conversation_size')
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':prompt},{'role':'assistant','content':patch}]
    return {'messages':messages,'language':language,'task_id':row['instance_id'],'repository':repo,
        'repository_family':family(repo),'split':split_for(repo),'example_kind':'real_repository_patch',
        'verification':{'kind':'publisher_execution','fail_to_pass_count':len(row['FAIL_TO_PASS']),
            'pass_to_pass_count':len(row.get('PASS_TO_PASS') or []),'locally_replayed':False},
        'provenance':{'dataset':DATASET,'revision':REVISION,'source_row_sha256':digest(canonical(row)),
            'source_url':'https://huggingface.co/datasets/'+DATASET+'/tree/'+REVISION,
            'repository_url':'https://github.com/'+repo,'repo_license':row['license'],'base_commit':row['base_commit']},
        'paths':paths}


def patch_identity(row):
    # Deduplicate backports/cherry-picks even when file locations and line numbers differ.
    changed=[line for line in row['messages'][-1]['content'].splitlines()
             if line[:1] in ('+','-') and not line.startswith(('+++','---'))]
    return digest(row['language']+'\n'+'\n'.join(' '.join(line.split()) for line in changed))


def select(rows, per_language=1500, per_repository=50):
    queues=defaultdict(list);seen_ids=set();seen_patches=set()
    for row in sorted(rows,key=lambda r:(digest(r['task_id']),r['task_id'])):
        key=patch_identity(row)
        if row['task_id'] in seen_ids or key in seen_patches:continue
        seen_ids.add(row['task_id']);seen_patches.add(key)
        queues[(row['split'],row['language'],row['repository_family'])].append(row)
    output=[]
    for split in ('train','dev','test'):
        for language in ('go','typescript'):
            groups=[deque(v[:per_repository]) for k,v in sorted(queues.items()) if k[:2]==(split,language)]
            selected=[];limit=per_language if split=='train' else min(100,per_language)
            while groups and len(selected)<limit:
                for group in list(groups):
                    if len(selected)>=limit:break
                    selected.append(group.popleft())
                    if not group:groups.remove(group)
            output.extend(selected)
    return output


def write_bundle(folder, rows, binding):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    binding_path=folder/'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text())!=binding:raise ValueError('Dataset preparation binding changed')
    atomic(binding_path,canonical(binding)+'\n')
    manifests={}
    for split in ('train','dev','test'):
        selected=[r for r in rows if r['split']==split]
        content=''.join(canonical(row)+'\n' for row in selected)
        path=folder/(split+'.jsonl');meta=Path(str(path)+'.manifest.json')
        if meta.exists() and (not path.exists() or json.loads(meta.read_text())['dataset_sha256']!=file_sha256(path)):
            raise ValueError('Prepared dataset digest changed')
        atomic(path,content)
        manifest={'schema_version':1,'split':split,'dataset_sha256':file_sha256(path),'rows':len(selected),
            'languages':dict(Counter(r['language'] for r in selected)),
            'repositories':len({r['repository_family'] for r in selected}), 'verification':'publisher_execution',
            'binding_sha256':file_sha256(binding_path),'sources':[{'dataset':DATASET,'revision':REVISION,
                'file':SOURCE_FILE,'sha256':SOURCE_SHA256,'license':'CC-BY-4.0',
                'attribution':'SWE-rebench V2, Ibragim Badertdinov, Maksim Nekrashevich, Anton Shevtsov, Alexander Golubev (2026). arXiv:2602.23866'}],
            'limitations':['Publisher test evidence was not locally replayed.','Source excerpts can omit relevant repository context.',
                'Repository-name family splits do not identify every renamed fork or copied implementation.',
                'Exact source exclusions and 13-word overlap checks cannot prove absence of semantic or pretraining contamination.']}
        atomic(meta,canonical(manifest)+'\n');manifests[split]=manifest
    atomic(folder/'summary.json',canonical({'completed':True,'splits':manifests})+'\n')
    return manifests


def download(url,path,expected):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        partial=path.with_name(path.name+'.partial')
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','3','--max-time','1200',
            '--output',str(partial),url],check=True)
        if file_sha256(partial)!=expected:raise ValueError('Downloaded source integrity mismatch')
        partial.replace(path)
    if file_sha256(path)!=expected:raise ValueError('Pinned source integrity mismatch')
    return path


def benchmark_exclusions():
    from pyarrow.parquet import read_table
    texts=[];sources={}
    original=ROOT/'research/HumanEval.jsonl.gz'
    with gzip.open(original,'rt') as stream:
        texts.extend(json.loads(line)['prompt'] for line in stream if line.strip())
    sources['HumanEval']=file_sha256(original)
    metadata=json.loads((ROOT/'research/multipl-e-source.json').read_text())
    for language,info in metadata['files'].items():
        path=download(metadata['url']+'/resolve/'+metadata['revision']+'/'+info['file'],
            ROOT/'.cache/benchmarks/multipl-e-source'/(language+'.parquet'),info['sha256'])
        texts.extend(row['prompt'] for row in read_table(path,columns=['prompt']).to_pylist())
        sources[language]=file_sha256(path)
    result=set()
    for text in texts:result.update(shingles(text))
    return result,sources


def prepare(folder,source=None,per_language=1500,per_repository=50):
    from pyarrow.parquet import ParquetFile
    folder=Path(folder)
    source=download('https://huggingface.co/datasets/'+DATASET+'/resolve/'+REVISION+'/'+SOURCE_FILE,
        source or folder/'source.parquet',SOURCE_SHA256)
    exclusions,audits=benchmark_exclusions();rows=[];rejected=Counter();count=0
    for batch in ParquetFile(source).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            count+=1
            try:rows.append(convert(row,exclusions))
            except ValueError as error:rejected[str(error)]+=1
        print('DATA scanned='+str(count)+' eligible='+str(len(rows)),flush=True)
    chosen=select(rows,per_language,per_repository)
    binding={'schema_version':1,'recipe':'focused-repair-v1','source_sha256':SOURCE_SHA256,
        'revision':REVISION,'preparer_sha256':file_sha256(__file__),'audit_sources':audits,
        'split':'Repository-name families; deterministic 80/10/10 train/dev/test',
        'per_language_train_target':per_language,'per_repository_cap':per_repository,'source_rows':count,
        'eligible':dict(Counter(r['language'] for r in rows)),'excluded':dict(rejected)}
    manifests=write_bundle(folder,chosen,binding)
    # Private audit sidecar enables later replay without putting hidden tests into prompts.
    selected_ids={r['task_id'] for r in chosen if r['split']!='train'}
    replay=[]
    for batch in ParquetFile(source).iter_batches(batch_size=256):
        replay.extend(r for r in batch.to_pylist() if r['instance_id'] in selected_ids)
    atomic(folder/'reserved-replay.jsonl',''.join(canonical(r)+'\n' for r in replay))
    print('DATA_SUMMARY='+canonical(manifests),flush=True)
    return manifests


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source',type=Path)
    parser.add_argument('--per-language',type=int,default=1500);parser.add_argument('--per-repository',type=int,default=50)
    args=parser.parse_args()
    if not 1<=args.per_language<=2500 or not 1<=args.per_repository<=100:parser.error('Dataset limits out of range')
    prepare(args.output,args.source,args.per_language,args.per_repository)
