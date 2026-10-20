"""Broader recursive distillation: children execute public tests before returning patches."""
import ast
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import agent_curriculum as curriculum
from agent_pilot import recursive_success
from focused_data import shingles
import teacher_data
from qwen35_train import atomic_json, sha256

ROOT = Path(__file__).resolve().parent
INSTRUCTION = (
    'Use supplied public tests when available; otherwise create small checks from the task '
    'description before repairing it. Keep supplied tests unchanged. '
    'For the root task (context is a dictionary), delegate one repair using rlm_query. '
    'Send JSON with task, editable, and files including the selected editable source and '
    'test_visible.py. The child must execute public tests before returning its JSON patch. '
    'Apply the child patch, run public tests, and return it. If tests fail, use their actual '
    'failure to correct the code instead of repeating the same proposal. '
    'For a child (context is a JSON string), parse the request, create its supplied files in '
    '/workspace, reproduce the visible failure, implement and execute the repair, and only '
    'then return the patch through answer content. Do not delegate further. '
    'Use fresh Python processes for tests to avoid stale imports.\n')


def eligible(entry, reserved):
    words=shingles(entry['prompt']+' '+entry['answer'])
    return teacher_data.split_for(entry)=='train' and all(
        t['repository']!='teacher-'+entry['family'] and not words & shingles(
            t['prompt']+' '+' '.join(t.get('files',{}).values())) for t in reserved)


def task(entry, bad_source=None):
    if teacher_data.split_for(entry)!='train':
        raise ValueError('Only training function families may become lessons')
    tree=ast.parse(entry['answer'])
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    checks=ast.parse(entry['tests']).body
    if len(functions)!=1 or len(checks)<5 or any(not isinstance(n,ast.Assert) for n in checks):
        raise ValueError('Expected one function and at least five separate assertions')
    imports='\n'.join(ast.unparse(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)))
    header=imports+'\nfrom solver import '+functions[0].name+'\n'
    public=header+'\n'.join(ast.unparse(n) for n in checks[:2])+"\nprint('VISIBLE_CHECKS_PASSED')\n"
    private=header+'\n'.join(ast.unparse(n) for n in checks[2:])+'\n'
    if bad_source is None:
        stub=copy.deepcopy(tree)
        next(n for n in stub.body if isinstance(n,ast.FunctionDef)).body=[ast.Raise(
            exc=ast.Call(func=ast.Name(id='NotImplementedError',ctx=ast.Load()),args=[],keywords=[]),cause=None)]
        bad_source=ast.unparse(ast.fix_missing_locations(stub))+'\n'
    return dict(id=entry['id'],repository='teacher-'+entry['family'],split='train',
        prompt=entry['prompt']+curriculum.PUBLIC_INSTRUCTION,
        files={'solver.py':bad_source,'test_visible.py':public},editable=['solver.py'],
        checks=private,reference_patch={'solver.py':entry['answer']},provenance=entry['provenance'])


def actions(task, recursive, visible=True):
    plan=[]
    def add(code,kind='root'):
        plan.append(dict(kind=kind,response='```repl\n'+code+'\n```'))
    setup='' if visible else (
        "from pathlib import Path\ncontext['files']['test_visible.py'] = "+repr(task['files']['test_visible.py'])+"\n"
        "Path('/workspace/test_visible.py').write_text(context['files']['test_visible.py'])\n")
    add(setup+"print(context['task'])\n"
        "for name in context['editable']:\n    print(name,context['files'][name])\n"
        "print(context['files']['test_visible.py'])\n"+curriculum.public_test(False))
    apply=("from pathlib import Path\n"
           "for name,source in patch.items():\n    Path('/workspace',name).write_text(source)\n"
           +curriculum.public_test(True)+"answer['content'] = patch\nanswer['ready'] = True")
    if recursive:
        add("import json\nrequest = {'task':context['task'],'editable':context['editable'],\n"
            "    'files':{name:context['files'][name] for name in context['editable']+['test_visible.py']}}\n"
            "child_reply = rlm_query(json.dumps(request))\nprint(child_reply)")
        add("import json\nfrom pathlib import Path\nrequest = json.loads(context)\n"
            "print(request['task'])\n"
            "for name,source in request['files'].items():\n"
            "    Path('/workspace',name).write_text(source)\n    print(name,source)\n"
            +curriculum.public_test(False),'subcall')
        add('patch = '+repr(task['reference_patch'])+'\n'+apply,'subcall')
        add('import json\npatch = json.loads(child_reply)\n'+apply)
    else:
        add('patch = '+repr(task['reference_patch'])+'\n'+apply)
    return plan


def failed_start(value):
    patch={p:value['files'][p] for p in value['editable']}
    private=curriculum.agent.grade(value,patch)
    public=curriculum.agent.grade(dict(value,checks=value['files']['test_visible.py']),patch)
    return not private['passed'] and not public['passed']


def prepare(folder, evaluation):
    from coding_loop_stage import progress
    folder.mkdir(parents=True,exist_ok=True)
    saved=teacher_data.prepare_round(folder.parent,folder,128,0,progress)
    config=json.loads(evaluation.read_text())
    reserved=[e['task'] for suite in ('development','holdout') for e in config[suite]]+config['retention']
    entries=[entry for entry in saved['train'] if eligible(entry,reserved)]
    if len(entries)<64 or len({e['family'] for e in entries})!=len(entries):
        raise ValueError('Insufficient distinct training families after evaluation exclusions')
    atomic_json(folder/'source.json',entries)
    atomic_json(folder/'prepare.json',dict(completed=True,teacher_sha256=sha256(folder/'teacher.json'),
        source_sha256=sha256(folder/'source.json'),evaluation_sha256=sha256(evaluation),families=len(entries)))


def collect(folder, revision, anchor_path, adapter):
    from coding_loop_stage import server, ask, progress, REVISION
    from qwen35_server import validate_adapter
    from qwen35_train import MODEL
    validate_adapter(adapter,MODEL,REVISION)
    entries=json.loads((folder/'source.json').read_text())
    prepared=json.loads((folder/'prepare.json').read_text())
    if prepared['source_sha256']!=sha256(folder/'source.json'):
        raise ValueError('Prepared teacher functions changed')
    sources=('agent_tested.py','agent_curriculum.py','agent_pilot.py','recursive_agent.py','rlm_worker.py',
             'coding_loop_stage.py','teacher_data.py','qwen35_server.py','structured_output.py','qwen35_train.py')
    binding=dict(revision=revision,policy_model_revision=REVISION,
        policy_adapter_sha256=sha256(adapter/'adapter_model.safetensors'),
        sources={n:sha256(ROOT/n) for n in sources},anchors_sha256=sha256(anchor_path),
        source_sha256=sha256(folder/'source.json'),prepare_sha256=sha256(folder/'prepare.json'),
        limits=curriculum.LIMITS,instruction=INSTRUCTION)
    path=folder/'binding.json'
    if path.exists() and json.loads(path.read_text())!=binding:
        raise ValueError('Cannot resume changed collection sources or weights')
    atomic_json(path,binding)
    (folder/'proposals').mkdir(exist_ok=True);(folder/'traces').mkdir(exist_ok=True)
    with server(adapter,folder):
        for index,entry in enumerate(entries):
            target=folder/'proposals'/(entry['id']+'.json')
            if target.exists():continue
            progress(folder,'collect-proposals',index,len(entries),task=entry['id'])
            answer=ask(teacher_data.messages(teacher_data.task(entry,'train')))
            value=task(entry,answer)
            atomic_json(target,dict(source_sha256=hashlib.sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest(),
                answer=answer,eligible_failed_start=failed_start(value)))
    expected=[]
    for index,entry in enumerate(entries):
        proposal=json.loads((folder/'proposals'/(entry['id']+'.json')).read_text())
        if proposal['source_sha256']!=hashlib.sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest():
            raise ValueError('Student proposal belongs to a different source function')
        starts=[('stub',None)]+([('student',proposal['answer'])] if proposal['eligible_failed_start'] else [])
        for start,answer in starts:
            value=task(entry,answer);value['id']+='-'+start
            variants=[(mode,visible) for mode in (False,True) for visible in ((True,False) if start=='stub' else (True,))]
            for recursive,visible in variants:
                name=value['id']+('-recursive' if recursive else '-ordinary')+('-public' if visible else '-source')+'.json'
                expected.append(name)
                if (folder/'traces'/name).exists():continue
                progress(folder,'verify-trajectories',index,len(entries),task=name)
                try:
                    record=curriculum.trajectory(value,recursive,visible,revision,plan=actions(value,recursive,visible),
                        instruction=INSTRUCTION if recursive else '')
                except curriculum.TrajectoryOutputLimit as error:
                    record=dict(rows=[],result=error.result,executions=error.executions,
                        excluded_reason='Cell output limit exceeded; whole failed trajectory excluded')
                if recursive and not record.get('excluded_reason') and not recursive_success(record['result'],value):
                    raise ValueError('Child patch was not genuinely executed and applied')
                record.update(family=entry['family'],recursive=recursive,start=start,visible=visible)
                for row in record['rows']:
                    row['provenance']=dict(entry['provenance'],construction='executed_tested_reference_actions',
                        model_discovered=False,start=start,policy_adapter_sha256=binding['policy_adapter_sha256'])
                atomic_json(folder/'traces'/name,record)
    if set(expected)!={p.name for p in (folder/'traces').glob('*.json')}:
        raise ValueError('Unexpected training traces')
    atomic_json(folder/'collection.json',dict(completed=True,families=len(entries),
        proposals={p.name:sha256(p) for p in (folder/'proposals').glob('*.json')},
        traces={name:sha256(folder/'traces'/name) for name in expected}))
    subprocess.run([str(ROOT/'.cache/qwen35-env/bin/python'),str(ROOT/'agent_tested.py'),'finalize',
        '--folder',str(folder),'--revision',REVISION,'--anchors',str(anchor_path)],check=True)


def finalize(folder, revision, anchor_path, tokenizer=None):
    from qwen35_train import MODEL, encode_rows, validate_tokenizer
    path=folder/'collection.json'
    if not path.exists() or not json.loads(path.read_text()).get('completed'):
        raise ValueError('Require a complete collection before training')
    collection=json.loads(path.read_text())
    files={p.name:sha256(p) for p in (folder/'traces').glob('*.json')}
    if not files or files!=collection['traces']:
        raise ValueError('Collected traces changed or are missing')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=revision,local_files_only=True,trust_remote_code=False)
        validate_tokenizer(tokenizer)
    rows,selected,excluded,measured=[],[],[],[]
    for name in sorted(files):
        record=json.loads((folder/'traces'/name).read_text());trace=record['result']['trace']
        if record.get('excluded_reason'):
            if (record['rows'] or record['result'].get('passed')
                    or record['result'].get('error')!='RuntimeError: Cell output limit exceeded'
                    or record['result'].get('infrastructure_errors')):
                raise ValueError('Invalid output-limit exclusion evidence')
            excluded.append(dict(file=name,reason=record['excluded_reason']))
            continue
        targets=[c['messages']+[dict(role='assistant',content=c['response'])] for c in trace]
        if (targets!=[r['messages'] for r in record['rows']] or len(trace)!=record['result']['calls']
                or [c['kind'] for c in trace]!=[r['origin'] for r in record['rows']]):
            raise ValueError('Training targets no longer match the executed trace')
        _,counts=encode_rows(tokenizer,record['rows'],4096)
        if (counts['excluded_count'] or len(trace)>curriculum.LIMITS['calls']
                or sum(a['answer_tokens'] for a in counts['included'])>curriculum.LIMITS['output_tokens']):
            excluded.append(dict(file=name,reason='Whole trajectory exceeds runtime token budget'))
            continue
        rows.extend(record['rows']);selected.append(record);measured.extend(counts['included'])
    families=set.intersection(*({r['family'] for r in selected if r['start']=='stub'
        and r['recursive']==mode and r['visible']==visible} for mode in (False,True) for visible in (False,True)))
    if len(families)<64:
        raise ValueError('Fewer than 64 families have all four complete workflow variants')
    binding=json.loads((folder/'binding.json').read_text())
    if (any(sha256(ROOT/name)!=digest for name,digest in binding['sources'].items())
            or sha256(anchor_path)!=binding['anchors_sha256']
            or sha256(folder/'source.json')!=binding['source_sha256']
            or sha256(folder/'prepare.json')!=binding['prepare_sha256']
            or collection['proposals']!={p.name:sha256(p) for p in (folder/'proposals').glob('*.json')}):
        raise ValueError('Collection source or proposal changed')
    replay=curriculum.anchors(anchor_path)
    if len(replay)!=150:raise ValueError('Expected 150 verified multilingual retention anchors')
    rows.extend(replay)
    curriculum.write_dataset(folder,rows,dict(binding,traces=files,collection_sha256=sha256(path),
        construction='execution_tested_recursive_distillation',excluded_trajectories=excluded))
    _,counts=encode_rows(tokenizer,rows,4096)
    if counts['excluded_count']:raise ValueError('Selected lesson or anchor exceeds token budget')
    dataset=folder/'train.jsonl'
    result=dict(completed=True,passed=True,all_agent_actions_included=True,complete_trajectories=True,
        sources_unchanged=True,dataset_sha256=sha256(dataset),manifest_sha256=sha256(str(dataset)+'.manifest.json'),
        tokenizer_model=MODEL,tokenizer_revision=revision,included_count=len(rows),excluded_count=0,
        training_families=len(families),accepted_trajectories=len(selected),excluded_trajectories=excluded,
        student_start_trajectories=sum(r['start']=='student' for r in selected),
        max_action_tokens=max(a['answer_tokens'] for a in measured),max_context_tokens=max(a['tokens'] for a in measured))
    atomic_json(folder/'feasibility.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','collect','finalize'))
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path)
    parser.add_argument('--revision')
    parser.add_argument('--anchors',type=Path)
    parser.add_argument('--adapter',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.folder,args.evaluation)
    elif args.action=='collect':collect(args.folder,args.revision,args.anchors,args.adapter)
    else:print(json.dumps(finalize(args.folder,args.revision,args.anchors)))


if __name__=='__main__':main()
