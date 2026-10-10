"""Bounded local replay of reserved SWE-rebench repairs in pinned publisher images."""
import argparse
import base64
import json
from pathlib import Path
import re
import shlex
import subprocess
import uuid

from evaluation import bounded_run
from focused_data import atomic, canonical, digest, excerpts, file_sha256, split_for

DEFAULT_IDS = ('magefile__mage-61','alecthomas__kong-484','tomarrell__lbadd-65',
    'sql-formatter-org__sql-formatter-421','bradymholt__cronstrue-297','cinar__indicatorts-428')
MAX_IMAGE_BYTES = 20*1024**3


def publisher_image(value):
    if not re.fullmatch(r'docker\.io/swerebenchv2/[a-z0-9_.-]+:[a-zA-Z0-9_.-]+',value):
        raise ValueError('Only publisher instance images may be replayed')
    return value


def validate_patch(row, patch):
    match=re.fullmatch(r'\s*```(?:diff|patch)?\s*\n(.*?)```\s*',patch,re.S)
    patch=match[1] if match else patch
    language='go' if row['language']=='go' else 'typescript'
    _,allowed=excerpts(row['patch'],language)
    _,paths=excerpts(patch,language)
    if not set(paths)<=set(allowed):raise ValueError('Candidate changes unregistered files')
    return patch if patch.endswith('\n') else patch+'\n'


def controls_valid(buggy,gold):
    return (buggy.get('test_started') is True and gold.get('test_started') is True
        and buggy.get('exit_code') not in (None,0,125,126,127)
        and gold.get('exit_code')==0 and not buggy.get('output_limited') and not gold.get('output_limited'))


def resolve_image(row, folder, consumed):
    target=publisher_image(row['image_name']);path=folder/'image.json'
    if path.exists():
        info=json.loads(path.read_text())
        if info['source_tag']!=target:raise ValueError('Publisher image binding changed')
    else:
        result=subprocess.run(['docker','manifest','inspect','--verbose',target],capture_output=True,text=True,timeout=60,check=True)
        manifest=json.loads(result.stdout)
        if isinstance(manifest,list):
            manifest=next(x for x in manifest if x['Descriptor'].get('platform',{}).get('architecture')=='amd64')
        sha=manifest['Descriptor']['digest']
        if not re.fullmatch(r'sha256:[0-9a-f]{64}',sha):raise ValueError('Invalid publisher image digest')
        compressed=sum(layer['size'] for layer in manifest['SchemaV2Manifest']['layers'])
        if compressed>4*1024**3:raise ValueError('Publisher image exceeds compressed download budget')
        info={'source_tag':target,'digest':target.split(':')[0]+'@'+sha,'compressed_bytes':compressed}
        atomic(path,canonical(info)+'\n')
    inspect=subprocess.run(['docker','image','inspect',info['digest']],capture_output=True,text=True,timeout=30)
    if inspect.returncode:
        with (folder/'pull.log').open('wb') as log:
            subprocess.run(['docker','pull',info['digest']],stdout=log,stderr=subprocess.STDOUT,timeout=1200,check=True)
        inspect=subprocess.run(['docker','image','inspect',info['digest']],capture_output=True,text=True,timeout=30,check=True)
    image=json.loads(inspect.stdout)[0];size=image['Size']
    if consumed+size>MAX_IMAGE_BYTES:
        subprocess.run(['docker','image','rm',info['digest']],capture_output=True,timeout=60)
        raise ValueError('Publisher images exceed the 20 GiB experiment limit')
    info.update(image_id=image['Id'],size_bytes=size)
    atomic(path,canonical(info)+'\n');return info


def run_patch(row, image, patch, timeout=300):
    # No host bind mounts, network, privileges or writable image filesystem.
    repo=row['repo'].split('/')[-1]
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',repo):raise ValueError('Unsafe repository name')
    if not re.fullmatch(r'[0-9a-f]{40}',row['base_commit']):raise ValueError('Invalid base commit')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',image['image_id']):raise ValueError('Unpinned runtime image')
    command=row['install_config']['test_cmd']
    if not isinstance(command,str) or not 1<=len(command)<=4096:raise ValueError('Missing bounded publisher test command')
    def encoded(value):return base64.b64encode(value.encode()).decode()
    script='set -eu\numask 077\nmkdir -p /workspace/repo /tmp/home\n'
    script+='cp -R /'+repo+'/. /workspace/repo/\ncd /workspace/repo\n'
    script+='git -c safe.directory=/workspace/repo reset --hard '+row['base_commit']+' >/tmp/reset.log 2>&1\n'
    if patch:
        script+='printf %s '+shlex.quote(encoded(patch))+' | base64 -d > /tmp/candidate.patch\n'
        script+='git -c safe.directory=/workspace/repo apply --recount --ignore-space-change --whitespace=nowarn /tmp/candidate.patch\n'
    script+='printf %s '+shlex.quote(encoded(row['test_patch']))+' | base64 -d > /tmp/tests.patch\n'
    script+='git -c safe.directory=/workspace/repo apply --recount --ignore-space-change --whitespace=nowarn /tmp/tests.patch\n'
    script+='printf %s '+shlex.quote(encoded(command))+' | base64 -d > /tmp/run-tests.sh\n'
    script+='ulimit -f 32768\necho FOCUSED_TEST_STARTED\nset +e\nbash /tmp/run-tests.sh >/tmp/test.log 2>&1\ncode=$?\n'
    script+='tail -c 3000 /tmp/test.log\necho\necho FOCUSED_TEST_EXIT=$code\nexit $code\n'
    name='local-focused-'+uuid.uuid4().hex
    args=['docker','run','--rm','-i','--name',name,'--network','none','--read-only','--cap-drop=ALL',
        '--security-opt=no-new-privileges','--pids-limit=512','--memory=6g','--cpus=2','--user=0:0',
        '--tmpfs','/workspace:rw,exec,nosuid,size=3g','--tmpfs','/tmp:rw,exec,nosuid,size=512m',
        '-e','GOCACHE=/tmp/gocache','-e','GOPROXY=off','-e','GOMAXPROCS=2',
        '-e','npm_config_offline=true','--entrypoint','/bin/bash',image['image_id'],'-s']
    try:
        result=bounded_run(args,script,timeout=timeout)
    except subprocess.TimeoutExpired:
        result={'exit_code':None,'output_limited':False,'output':'Test timed out'}
    finally:
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=15)
    result['test_started']='FOCUSED_TEST_STARTED' in result['output']
    result['passed']=result['test_started'] and result['exit_code']==0 and not result['output_limited']
    return result


def prepare(data, output, ids=DEFAULT_IDS):
    data,output=Path(data),Path(output);output.mkdir(parents=True,exist_ok=True)
    rows={r['instance_id']:r for r in map(json.loads,data.read_text().splitlines())}
    binding={'schema_version':1,'data_sha256':file_sha256(data),'task_ids':list(ids),
        'replay_sha256':file_sha256(__file__),'phase':'reserved_development_controls','timeout_seconds':300}
    previous=output/'binding.json'
    if previous.exists() and json.loads(previous.read_text())!=binding:raise ValueError('Replay preparation binding changed')
    atomic(previous,canonical(binding)+'\n');selected=[];consumed=0;failures=[]
    for task_id in ids:
        row=rows[task_id]
        if split_for(row['repo'])!='dev':raise ValueError('Replay pilot only consumes the development split')
        folder=output/task_id;folder.mkdir(exist_ok=True)
        print('REPLAY preparing '+task_id,flush=True)
        try:
            image=resolve_image(row,folder,consumed);consumed+=image['size_bytes']
            results={}
            for mode,patch in [('buggy',''),('gold',row['patch'])]:
                path=folder/(mode+'.json')
                run_binding={'source_row_sha256':digest(canonical(row)),'image_id':image['image_id'],'mode':mode,
                    'patch_sha256':digest(patch),'replay_sha256':binding['replay_sha256']}
                if path.exists():
                    value=json.loads(path.read_text())
                    if value['binding']!=run_binding:raise ValueError('Reserved replay control binding changed')
                    results[mode]=value['result']
                else:
                    print('REPLAY '+task_id+' '+mode,flush=True)
                    results[mode]=run_patch(row,image,patch)
                    atomic(path,canonical({'binding':run_binding,'result':results[mode]})+'\n')
            if controls_valid(results['buggy'],results['gold']):
                selected.append({'task_id':task_id,'source_row_sha256':digest(canonical(row)),'image':image,
                    'controls':{'buggy_failed':True,'gold_passed':True},'language':row['language']})
            else:failures.append({'task_id':task_id,'reason':'local_controls_failed','results':results})
        except (ValueError,subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            failures.append({'task_id':task_id,'reason':str(error)[-2000:]})
        atomic(output/'selection.json',canonical({'schema_version':1,'completed':False,'binding':binding,
            'tasks':selected,'failures':failures,'image_size_bytes':consumed})+'\n')
    value={'schema_version':1,'completed':True,'binding':binding,'tasks':selected,'failures':failures,'image_size_bytes':consumed}
    atomic(output/'selection.json',canonical(value)+'\n');print('REPLAY_READY='+str(len(selected)),flush=True)
    return value


def grade(data, selection, task_id, answer):
    data,selection=Path(data),Path(selection);index=json.loads(selection.read_text())
    if (not index.get('completed') or index['binding']['data_sha256']!=file_sha256(data)
            or index['binding']['replay_sha256']!=file_sha256(__file__)):
        raise ValueError('Replay selection is incomplete or changed')
    chosen=next(t for t in index['tasks'] if t['task_id']==task_id)
    row=next(r for r in map(json.loads,data.read_text().splitlines()) if r['instance_id']==task_id)
    if digest(canonical(row))!=chosen['source_row_sha256']:raise ValueError('Reserved source row changed')
    try:patch=validate_patch(row,answer)
    except ValueError as error:return {'passed':False,'exit_code':None,'feedback':str(error),'kind':'invalid_patch'}
    result=run_patch(row,chosen['image'],patch)
    return {**result,'feedback':result['output'],'kind':'local_repository_tests'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ids',nargs='+',default=DEFAULT_IDS)
    args=parser.parse_args();prepare(args.data,args.output,args.ids)
