"""Executed multi-file lessons: derive checks from the specification, never invent supplied tests."""
import argparse
import ast
import json
from pathlib import Path
import subprocess

import agent_curriculum as curriculum
import agent_tested
from qwen35_train import atomic_json, sha256

ROOT = Path(__file__).resolve().parent
ORDINARY = ('Inspect the file inventory and the relevant imports, including supporting files. '
    'No test file is guaranteed to exist. If none is supplied, write a small check script from '
    'the task requirements and examples, execute it, repair the source, and execute it again '
    'in a fresh Python process. Keep context unchanged and keep existing tests unchanged. '
    'Create parent directories when writing files. Include every required source edit in the '
    'final JSON patch; do not include scratch tests. Use actual failures to revise the repair.\n')
INSTRUCTION = ORDINARY + ('For a root task (context is a dictionary), delegate one focused repair '
    'with rlm_query after inspection. Send JSON containing task, editable, and relevant files '
    'including imported supporting modules. Do not assume a test filename or omit dependencies. '
    'For a child (context is a JSON string), parse it and write the supplied files under /workspace. '
    'Derive your own checks if no tests are supplied, execute them before and after repair, '
    'then return the JSON patch through answer content. Do not delegate further. '
    'The parent must parse the actual child response, apply it, run its own checks, and finish.\n')
SOURCES = ('agent_multifile.py','agent_curriculum.py','agent_tested.py','agent_pilot.py',
    'recursive_agent.py','rlm_worker.py','qwen35_train.py','qwen35_server.py','structured_output.py','teacher_data.py')


def task(entry, index):
    # Verified function examples supply diagnostics; the remaining checks stay private.
    original = agent_tested.task(entry)
    tree = ast.parse(entry['answer'])
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef)).name
    package = ('engine','libcode','components','domain')[index % 4]
    module = ('operations','compute','logic','rules')[index % 4]
    gateway = ('service','api','application','facade')[index % 4]
    core = f'{package}/{module}.py'
    entrypoint = gateway+'.py'
    utility = package+'/bridge.py'
    wrapper = (f'from {package}.{module} import {function} as implementation\n'
        f'from {package}.bridge import invoke\n\ndef {function}(*args, **kwargs):\n'
        '    return invoke(implementation, args, kwargs)\n')
    tests = ast.parse(entry['tests']).body
    imports = '\n'.join(ast.unparse(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)))
    header = imports+f'\nfrom {gateway} import {function}\n'
    examples = '\n'.join(ast.unparse(n) for n in tests[:2])
    return dict(id=entry['id']+'-repository',repository='teacher-'+entry['family'],split='train',
        prompt=(entry['prompt']+f' The public API is {gateway}.{function}; it must use the '
            f'implementation in {package}.{module} through the provided bridge. Repair both '
            'the implementation and its caller. No tests are supplied. Required examples:\n'+examples),
        files={core:original['files']['solver.py'],entrypoint:wrapper.replace(
            'return invoke(implementation, args, kwargs)','return None'),package+'/__init__.py':'',
            utility:'def invoke(function, args, kwargs):\n    return function(*args, **kwargs)\n'},
        editable=[core,entrypoint],reference_patch={core:entry['answer'],entrypoint:wrapper},
        checks=header+'\n'.join(ast.unparse(n) for n in tests[2:])+'\n',
        diagnostic=header+examples+"\nprint('ISSUE_CHECKS_PASSED')\n",provenance=entry['provenance'])


def check_code(passed):
    program="import sys;sys.path.insert(0,'/workspace');exec(open('/workspace/check_issue.py').read())"
    return ("import subprocess,sys\n"
        f"tested = subprocess.run([sys.executable,'-B','-I','-c',{program!r}],capture_output=True,text=True,timeout=45)\n"
        "print('Check exit:',tested.returncode)\nprint((tested.stdout+tested.stderr)[-1800:])\n"
        + ("assert tested.returncode == 0 and 'ISSUE_CHECKS_PASSED' in tested.stdout\n" if passed
           else "assert tested.returncode != 0\n"))


def actions(value, recursive):
    plan=[]
    def add(code,kind='root'):plan.append(dict(kind=kind,response='```repl\n'+code+'\n```'))
    add("print(context['task'])\nprint(list(context['files']))\n"
        "for name,source in context['files'].items():\n    print(name,source[:1200])")
    diagnostic=("from pathlib import Path\nPath('/workspace/check_issue.py').write_text("
        +repr(value['diagnostic'])+")\n")
    apply=("from pathlib import Path\n"
        "for name,source in patch.items():\n"
        "    path=Path('/workspace',name)\n    path.parent.mkdir(parents=True,exist_ok=True)\n"
        "    path.write_text(source)\n"+check_code(True)+"answer['content']=patch\nanswer['ready']=True")
    if recursive:
        add(diagnostic+check_code(False)+"import json\n"
            "request={'task':context['task'],'editable':context['editable'],'files':context['files']}\n"
            "child_reply=rlm_query(json.dumps(request))\nprint(child_reply)")
        add("import json\nfrom pathlib import Path\nrequest=json.loads(context)\n"
            "print(request['task'])\n"
            "for name,source in request['files'].items():\n"
            "    path=Path('/workspace',name)\n    path.parent.mkdir(parents=True,exist_ok=True)\n"
            "    path.write_text(source)\n"+diagnostic+check_code(False),'subcall')
        add('patch='+repr(value['reference_patch'])+'\n'+apply,'subcall')
        add('import json\npatch=json.loads(child_reply)\n'+apply)
    else:
        add(diagnostic+check_code(False))
        add('patch='+repr(value['reference_patch'])+'\n'+apply)
    return plan


def mode_gains(before, after):
    if not before or [(r['id'],r['condition']) for r in before]!=[(r['id'],r['condition']) for r in after]:
        raise ValueError('Require complete matching ordered tasks')
    if len({(r['id'],r['condition']) for r in before}) != len(before):
        raise ValueError('Require unique task and mode pairs')
    for row in before+after:
        if (row['condition'] not in ('ordinary','recursive')
                or any(type(row.get(k)) is not bool for k in ('passed','recursive_passed','repl_passed'))
                or (row['recursive_passed'] or row['repl_passed']) and not row['passed']):
            raise ValueError('Require boolean completed scores')
    ordinary=[a['id'] for b,a in zip(before,after) if a['condition']=='ordinary'
              and a['repl_passed'] and not b['repl_passed']]
    recursive=[a['id'] for b,a in zip(before,after) if a['condition']=='recursive' and a['recursive_passed'] and not b['recursive_passed']]
    lost=[a['id'] for b,a in zip(before,after) if b['passed'] and not a['passed']
          or b['recursive_passed'] and not a['recursive_passed']
          or b['repl_passed'] and not a['repl_passed']]
    return dict(passed=bool(ordinary and recursive and not lost),ordinary_gains=ordinary,recursive_gains=recursive,losses=lost)


def lesson_rejection(value):
    """Only lessons whose independent checks distinguish both repairs may teach actions."""
    reference=value['reference_patch']
    broken={p:value['files'][p] for p in value['editable']}
    if not curriculum.agent.grade(value,reference)['passed']:
        return 'reference fails private checks'
    if curriculum.agent.grade(value,broken)['passed']:
        return 'private checks accept the broken caller'
    diagnostic=dict(value,checks=value['diagnostic'])
    if not curriculum.agent.grade(diagnostic,reference)['passed']:
        return 'reference fails specification examples'
    if curriculum.agent.grade(diagnostic,broken)['passed']:
        return 'specification examples accept the broken caller'
    for omitted in value['editable']:
        partial={p:s for p,s in reference.items() if p!=omitted}
        if curriculum.agent.grade(value,partial)['passed']:
            return 'private checks do not require both source repairs'
        if curriculum.agent.grade(diagnostic,partial)['passed']:
            return 'specification examples do not require both source repairs'
    return None


def prepare(folder, source, evaluation):
    from coding_loop_stage import progress
    entries=json.loads(source.read_text());config=json.loads(evaluation.read_text())
    reserved=[e['task'] for suite in ('development','holdout') for e in config[suite]]+config['retention']
    entries=[e for e in entries if agent_tested.eligible(e,reserved)]
    if len(entries)<64:raise ValueError('Too few disjoint train-only families')
    folder.mkdir(parents=True,exist_ok=True)
    values=[];excluded=[]
    for i,entry in enumerate(entries):
        value=task(entry,i)
        progress(folder,'prepare',i,len(entries),task=value['id'])
        reason=lesson_rejection(value)
        if reason:excluded.append(dict(id=value['id'],reason=reason))
        else:values.append(value)
    atomic_json(folder/'excluded-lessons.json',excluded)
    if len(values)<64:raise ValueError('Fewer than 64 lessons have discriminating checks')
    atomic_json(folder/'tasks.json',values)
    atomic_json(folder/'prepare.json',dict(completed=True,source_sha256=sha256(source),
        evaluation_sha256=sha256(evaluation),tasks_sha256=sha256(folder/'tasks.json'),families=len(values),
        excluded_sha256=sha256(folder/'excluded-lessons.json'),excluded_count=len(excluded)))


def collect(folder, revision, anchors):
    from coding_loop_stage import progress, REVISION
    values=json.loads((folder/'tasks.json').read_text())
    prepared=json.loads((folder/'prepare.json').read_text())
    if sha256(folder/'tasks.json')!=prepared['tasks_sha256']:raise ValueError('Prepared tasks changed')
    binding=dict(sources={n:sha256(ROOT/n) for n in SOURCES},tasks_sha256=sha256(folder/'tasks.json'),
        prepare_sha256=sha256(folder/'prepare.json'),anchors_sha256=sha256(anchors),revision=revision,
        construction='multifile_spec_check_distillation',instruction=INSTRUCTION,ordinary_instruction=ORDINARY)
    if (folder/'binding.json').exists() and json.loads((folder/'binding.json').read_text())!=binding:
        raise ValueError('Cannot resume changed collection')
    atomic_json(folder/'binding.json',binding);(folder/'traces').mkdir(exist_ok=True)
    expected=[]
    for i,value in enumerate(values):
        for recursive in (False,True):
            name=value['id']+('-recursive' if recursive else '-ordinary')+'.json';expected.append(name)
            if (folder/'traces'/name).exists():continue
            progress(folder,'collect',i,len(values),task=name)
            try:
                record=curriculum.trajectory(value,recursive,False,revision,plan=actions(value,recursive),
                    instruction=INSTRUCTION if recursive else ORDINARY,mode='rlm' if recursive else 'baseline')
            except curriculum.TrajectoryOutputLimit as error:
                record=dict(rows=[],result=error.result,executions=error.executions,excluded_reason='output_limit')
            record.update(family=value['repository'],recursive=recursive)
            atomic_json(folder/'traces'/name,record)
    files={p.name:sha256(p) for p in (folder/'traces').glob('*.json')}
    if set(files)!=set(expected):raise ValueError('Unexpected or missing trace')
    atomic_json(folder/'collection.json',dict(completed=True,traces=files))
    subprocess.run([str(ROOT/'.cache/qwen35-env/bin/python'),str(ROOT/'agent_multifile.py'),
        'finalize','--folder',str(folder),'--revision',REVISION,'--anchors',str(anchors)],check=True)


def finalize(folder, revision, anchors):
    from transformers import AutoTokenizer
    from qwen35_train import MODEL, encode_rows, validate_tokenizer
    binding=json.loads((folder/'binding.json').read_text())
    collection=json.loads((folder/'collection.json').read_text())
    if (not collection['completed'] or any(sha256(ROOT/n)!=h for n,h in binding['sources'].items())
            or sha256(anchors)!=binding['anchors_sha256'] or sha256(folder/'tasks.json')!=binding['tasks_sha256']):
        raise ValueError('Collection changed')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=revision,local_files_only=True,trust_remote_code=False)
    validate_tokenizer(tokenizer)
    rows=[];excluded=[];selected=[]
    for name,digest in sorted(collection['traces'].items()):
        path=folder/'traces'/name
        if sha256(path)!=digest:raise ValueError('Trace changed')
        record=json.loads(path.read_text());result=record['result']
        if record.get('excluded_reason'):
            if record['rows'] or result.get('passed') or result.get('infrastructure_errors') or result.get('error')!='RuntimeError: Cell output limit exceeded':
                raise ValueError('Invalid failed trace exclusion')
            excluded.append(dict(file=name,reason='output_limit'));continue
        targets=[c['messages']+[dict(role='assistant',content=c['response'])] for c in result['trace']]
        indices=record.get('target_indices',list(range(len(targets))))
        if (not indices or indices!=sorted(set(indices)) or any(i<0 or i>=len(targets) for i in indices)
                or [targets[i] for i in indices]!=[r['messages'] for r in record['rows']]
                or len(targets)!=result['calls']):
            raise ValueError('Unexecuted training target')
        full_trace=[dict(record['rows'][0],task_id=f'preflight-{i}',messages=t) for i,t in enumerate(targets)]
        _,counts=encode_rows(tokenizer,full_trace,4096)
        limits=binding.get('limits',curriculum.LIMITS)
        if (counts['excluded_count'] or result['calls']>limits['calls']
                or sum(r['answer_tokens'] for r in counts['included'])>limits['output_tokens']):
            excluded.append(dict(file=name,reason='whole_trajectory_token_budget'));continue
        rows.extend(record['rows']);selected.append(record)
    families=set.intersection(*({r['family'] for r in selected if r['recursive']==mode} for mode in (False,True)))
    if len(families)<48:raise ValueError('Fewer than 48 complete ordinary/recursive family pairs')
    replay=curriculum.anchors(anchors)
    if len(replay)!=150:raise ValueError('Missing multilingual retention anchors')
    rows.extend(replay)
    curriculum.write_dataset(folder,rows,dict(binding,traces=collection['traces'],excluded=excluded))
    _,counts=encode_rows(tokenizer,rows,4096)
    if counts['excluded_count']:raise ValueError('Selected dataset truncates actions')
    atomic_json(folder/'feasibility.json',dict(completed=True,passed=True,all_agent_actions_included=True,
        complete_trajectories=True,sources_unchanged=True,dataset_sha256=sha256(folder/'train.jsonl'),
        manifest_sha256=sha256(folder/'train.jsonl.manifest.json'),tokenizer_model=MODEL,tokenizer_revision=revision,
        included_count=len(rows),excluded_count=0,training_families=len(families),accepted_trajectories=len(selected),
        excluded_trajectories=excluded,ordinary_trajectories=sum(not r['recursive'] for r in selected),
        recursive_trajectories=sum(r['recursive'] for r in selected)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','collect','finalize'))
    for name in ('folder','source','evaluation','anchors'):p.add_argument('--'+name,type=Path,required=name=='folder')
    p.add_argument('--revision');a=p.parse_args()
    if a.action=='prepare':prepare(a.folder,a.source,a.evaluation)
    elif a.action=='collect':collect(a.folder,a.revision,a.anchors)
    else:finalize(a.folder,a.revision,a.anchors)


if __name__=='__main__':main()
