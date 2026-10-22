"""Short executed REPL actions with masked failures and explicit recovery."""
import argparse
import json
from pathlib import Path
import subprocess

import agent_curriculum as curriculum
import agent_multifile as multi
from qwen35_train import atomic_json, sha256

ROOT = Path(__file__).resolve().parent
LIMITS = dict(depth=2, calls=12, output_tokens=8192, seconds=360)
SOURCES = multi.SOURCES + ('agent_atomic.py', 'repl_actions.py')


def actions(value, recursive, recovery):
    if recovery not in ('syntax', 'semantic', 'missing_edit'):
        raise ValueError('Unknown recovery lesson')
    plan = []
    def add(code, kind='root', **extra):
        plan.append(dict(kind=kind, response='```repl\n'+code+'\n```', **extra))
    core, caller = value['editable']
    reference = value['reference_patch']
    check = 'check('+repr(value['diagnostic'])+')'
    kind = 'subcall' if recursive else 'root'
    if recursive:
        add('show()')
        add('proposal = delegate("Repair the public API across its implementation and caller. '
            'Derive assertions from the requirements, run them before and after editing, and finish.")')
    add('show()', kind)
    add(check, kind)
    if recovery == 'syntax':
        add('write('+repr(core)+', '+repr('def incomplete(:\n    pass\n')+')',
            kind, train=False, expected_error='SyntaxError')
    elif recovery == 'semantic':
        add('write('+repr(core)+', '+repr(value['files'][core])+')', kind, train=False)
        add('write('+repr(caller)+', '+repr(reference[caller])+')', kind)
        add(check, kind)
    add('write('+repr(core)+', '+repr(reference[core])+')', kind)
    if recovery == 'missing_edit':
        add(check, kind)
    if recovery != 'semantic':
        add('write('+repr(caller)+', '+repr(reference[caller])+')', kind)
    add(check+'\nfinish()', kind)
    if recursive:
        add('apply(proposal)\n'+check+'\nfinish()')
    return plan


def verify_record(record, value):
    from agent_pilot import recursive_success
    result = record['result']
    plan = actions(value, record['recursive'], record['recovery'])
    expected = [i for i,a in enumerate(plan) if a.get('train') is not False]
    if record['target_indices'] != expected:
        raise ValueError('Failed action mask changed')
    if ([r['messages'] for r in record['rows']] != [
            c['messages']+[dict(role='assistant',content=c['response'])]
            for i,c in enumerate(result['trace']) if i in expected]
            or [c['response'] for c in result['trace']] != [a['response'] for a in plan]):
        raise ValueError('Lesson targets differ from executed actions')
    if (not result['passed'] or result.get('error') or result.get('infrastructure_errors')
            or result['patch'] != value['reference_patch']):
        raise ValueError('Recovery did not produce a verified patch')
    executions = result['executions']
    if record['recursive']:
        if not recursive_success(result, value):
            raise ValueError('No genuine tested child repair')
        query = result['queries'][0]
        executions = executions[query['execution_start']:query['execution_end']]
    elif result['queries']:
        raise ValueError('Ordinary lesson delegated')
    checks = [c for e in executions for c in e['checks']]
    expected_failures = 1 if record['recovery']=='syntax' else 2
    if (len(checks) != expected_failures+1 or any(not c['assertions'] for c in checks)
            or any(c['returncode']==0 for c in checks[:-1]) or checks[-1]['returncode']!=0):
        raise ValueError('Recovery checks did not fail then pass')


def collect(folder, revision, anchors):
    from coding_loop_stage import progress, REVISION
    values=json.loads((folder/'tasks.json').read_text())
    prepared=json.loads((folder/'prepare.json').read_text())
    if sha256(folder/'tasks.json')!=prepared['tasks_sha256']:
        raise ValueError('Prepared tasks changed')
    binding=dict(sources={n:sha256(ROOT/n) for n in SOURCES},tasks_sha256=sha256(folder/'tasks.json'),
        prepare_sha256=sha256(folder/'prepare.json'),anchors_sha256=sha256(anchors),revision=revision,
        construction='short_action_recovery_imitation',action_api=True,limits=LIMITS)
    if (folder/'binding.json').exists() and json.loads((folder/'binding.json').read_text())!=binding:
        raise ValueError('Cannot resume changed collection')
    atomic_json(folder/'binding.json',binding)
    (folder/'traces').mkdir(exist_ok=True)
    expected=[]
    for i,value in enumerate(values):
        recovery=('syntax','semantic','missing_edit')[i%3]
        for recursive in (False,True):
            name=value['id']+('-recursive' if recursive else '-ordinary')+'.json'
            expected.append(name)
            if (folder/'traces'/name).exists():
                continue
            progress(folder,'collect',i,len(values),task=name)
            try:
                record=curriculum.trajectory(value,recursive,False,revision,
                    plan=actions(value,recursive,recovery),mode='rlm' if recursive else 'baseline',
                    action_api=True,limits=LIMITS)
                record.update(recursive=recursive,recovery=recovery)
                verify_record(record,value)
            except curriculum.TrajectoryOutputLimit as error:
                record=dict(rows=[],result=error.result,executions=error.executions,excluded_reason='output_limit')
            record.update(family=value['repository'],recursive=recursive,recovery=recovery)
            atomic_json(folder/'traces'/name,record)
    files={p.name:sha256(p) for p in (folder/'traces').glob('*.json')}
    if set(files)!=set(expected):
        raise ValueError('Unexpected or missing trace')
    atomic_json(folder/'collection.json',dict(completed=True,traces=files))
    subprocess.run([str(ROOT/'.cache/qwen35-env/bin/python'),str(ROOT/'agent_atomic.py'),
        'finalize','--folder',str(folder),'--revision',REVISION,'--anchors',str(anchors)],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','collect','finalize'))
    for name in ('folder','source','evaluation','anchors'):
        p.add_argument('--'+name,type=Path,required=name=='folder')
    p.add_argument('--revision')
    a=p.parse_args()
    if a.action=='prepare':
        multi.prepare(a.folder,a.source,a.evaluation)
    elif a.action=='collect':
        collect(a.folder,a.revision,a.anchors)
    else:
        multi.finalize(a.folder,a.revision,a.anchors)


if __name__=='__main__':
    main()
