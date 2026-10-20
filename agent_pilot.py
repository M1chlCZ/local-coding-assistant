"""Finite, matched REPL curriculum pilot. A failed gate never starts a learning loop."""
import argparse
import hashlib
import json
import socket
import signal
import subprocess
import time
from pathlib import Path

from qwen35_train import atomic_json, sha256
from focused_experiment import ROOT, read
from recursive_agent import tested_repl
from repl_learning import LIMITS, run_child

RECURSIVE = ('For the root repository task (context is a dictionary), '
             'use one focused rlm_query to obtain a proposed repair from a child REPL. '
             'Pass the task and selected editable source. Ask the child for a JSON patch. '
             'Apply that patch in the parent workspace, run available public tests, and return it. '
             'For a child subtask (context is a string), analyze only the supplied source and return '
             'the requested JSON patch; no further delegation or repository filesystem is required.\n')
SOURCES = ('recursive_agent.py', 'rlm_worker.py', 'repl_actions.py', 'evaluation.py', 'qwen35_server.py', 'structured_output.py',
           'qwen35_train.py', 'agent_pilot.py', 'coding_loop_stage.py', 'repl_learning.py',
           'code_recipe.py', 'polyglot_runtime.py', 'research/adapter_eval_server.py')


def source_hashes():
    from rlm.core import rlm
    from rlm.utils import parsing
    return {**{name: sha256(ROOT/name) for name in SOURCES},
            'upstream_rlm':sha256(rlm.__file__), 'upstream_parsing':sha256(parsing.__file__)}


def freeze(folder, retention):
    from coding_loop_stage import REVISION
    from recursive_agent import UPSTREAM
    if (folder/'evaluation.json').exists():
        raise FileExistsError('Evaluation already frozen; reuse it unchanged')
    authored = read(ROOT/'research/repair_tasks.json')
    original = read(ROOT/'research/rlm_tasks.json')
    known = {t['id']: t for t in authored + original}
    ordinary = ['matrix-transpose', 'date-span', 'bounded-tail', 'nested-update']
    recursive = ['isbn-check', 'roman-numerals', 'edit-distance', 'header-multimap']
    entries = [dict(task=known['tuning-'+name], condition=condition) for condition, names in
               [('ordinary', ordinary), ('recursive', recursive)] for name in names]
    holdout = [dict(task=t, condition='recursive' if i >= 3 else 'ordinary')
               for i, t in enumerate(t for t in original if t['split']=='holdout')]
    development = read(retention)['validation']
    anchors = [t for language in ('python','go','typescript','rust','dart')
               for t in [t for t in development if t.get('language')==language][:3]]
    train_families = {t['repository'] for t in authored+original if t['split']=='train'}
    assert len(anchors)==15 and len(entries)==8 and len(holdout)==5
    assert not train_families & {e['task']['repository'] for e in entries+holdout}
    assert all(e['task']['split']!='train' for e in entries+holdout)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder/'evaluation.json', dict(schema=1, model_revision=REVISION,
        upstream=UPSTREAM, limits=LIMITS, instruction=RECURSIVE, sources=source_hashes(),
        development=entries, holdout=holdout, retention=anchors,
        gate=dict(distinct_gains=2, recursive_gains=1, max_losses=0,
                  holdout_gains=1, max_recipes=2),
        recipes=[dict(learning_rate=rate,max_epochs=3,max_steps=200,rank=8,seed=3407,
                      max_length=4096,gradient_accumulation=8) for rate in (5e-5,1e-4)],
        frozen_at=time.time()))


def recursive_success(row, task):
    from recursive_agent import parse_patch
    if not row.get('passed') or not row.get('patch'):
        return False
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name,source in row['patch'].items()}
    if row.get('workspace_hashes') != hashes:
        return False
    for query in row.get('queries', []):
        if not query.get('recursive'):
            continue
        try:
            proposal = parse_patch(query['response'], task)
            executions = row['executions'][query['execution_start']:query['execution_end']]
            completed = any(not e['stderr'] and e['final_answer'] is not None
                            and parse_patch(e['final_answer'],task)==proposal for e in executions)
            checked = True
            if row.get('action_api'):
                checks = [c for e in executions for c in e.get('checks', [])]
                checked = bool(checks and any(c['returncode'] != 0 and c['assertions'] for c in checks)
                    and checks[-1]['returncode'] == 0 and checks[-1]['assertions']
                    and checks[-1]['before'] == checks[-1]['after']
                    and all(checks[-1]['after'].get(p)==h for p,h in hashes.items()) and tested_repl(row))
            if completed and checked and proposal==row['patch']:
                return True
        except (ValueError, KeyError, TypeError):
            continue
    return False


def evaluate_one(entry):
    from recursive_agent import solve, grade
    from polyglot_runtime import check_program
    if not check_program('python','print(1)')['passed']:
        raise RuntimeError('Independent runtime check failed')
    task=entry['task']
    limits=entry.get('limits',LIMITS)
    row=solve(task,'http://127.0.0.1:8090',
              mode='baseline' if entry.get('strict_ordinary') and entry['condition']=='ordinary' else 'rlm',
              instruction=entry.get('instruction',RECURSIVE) if entry['condition']=='recursive'
              else entry.get('ordinary_instruction',''), action_api=entry.get('action_api',False),
              context_window=entry.get('context_window',False), **limits)
    if row.get('infrastructure_errors'):
        raise RuntimeError('Infrastructure failure: '+'; '.join(row['infrastructure_errors']))
    error=row.get('error','')
    expected=('ValueError:', 'JSONDecodeError:', 'TimeoutError:', 'TimeoutExceededError:',
              'ErrorThresholdExceededError:', 'RuntimeError: Model call or output-token budget exhausted',
              'RuntimeError: Cell output limit exceeded')
    expected += ('ContextWindowExceeded:',)
    if error and not error.startswith(expected) and not (
            error.startswith('APITimeoutError:') and row['elapsed_s']>=limits['seconds']-1):
        raise RuntimeError('Infrastructure failure: '+error)
    if row.get('patch'):
        row.update(grade(task,row['patch']))
    row.update(condition=entry['condition'], recursive_passed=recursive_success(row,task))
    row['repl_passed']=tested_repl(row)
    return row


def evaluate(folder, label, adapter, suite):
    from coding_loop_stage import server, task_messages, ask, check_task
    from launcher import health
    config=read(folder/'evaluation.json')
    if config['sources'] != source_hashes():
        raise ValueError('Pilot evaluation source changed')
    target=folder/label; target.mkdir(exist_ok=True)
    path=target/(suite+'.json')
    binding=dict(evaluation_sha256=sha256(folder/'evaluation.json'),
                 adapter_sha256=sha256(adapter/'adapter_model.safetensors') if adapter else None, suite=suite)
    saved=read(path,dict(binding=binding,rows=[],retention=[]))
    if saved['binding']!=binding:
        raise ValueError('Pilot weights or evaluation changed')
    entries=config[suite]
    if len(saved['rows'])>len(entries):raise ValueError('Unexpected extra results')
    with server(adapter,target):
        for entry in entries[len(saved['rows']):]:
            task=target/'task.json'; result=target/'task-result.json'
            atomic_json(task,dict(entry,instruction=config['instruction'],
                ordinary_instruction=config.get('ordinary_instruction',''),
                strict_ordinary=config.get('strict_ordinary',False),action_api=config.get('action_api',False),
                context_window=config.get('context_window',False),
                limits=config.get('limits',LIMITS))); result.unlink(missing_ok=True)
            run_child([str(ROOT/'.cache/rlm-env/bin/python'),str(ROOT/'agent_pilot.py'),
                       'task','--folder',str(target)],config.get('limits',LIMITS)['seconds']+180)
            row=read(result)
            if row['id']!=entry['task']['id']:raise ValueError('Task result mismatch')
            saved['rows'].append(row); atomic_json(path,saved)
            print(f"{label}/{suite}: {len(saved['rows'])}/{len(entries)} {row['id']} "
                  f"passed={row['passed']} recursive={row['recursive_passed']}",flush=True)
            deadline=time.monotonic()+60
            while not health(8090):
                if time.monotonic()>deadline:raise RuntimeError('Model did not recover after task')
                time.sleep(1)
        if suite=='development':
            for task in config['retention'][len(saved['retention']):]:
                answer=ask(task_messages(task)); checked=check_task(task,answer)
                saved['retention'].append(dict(id=task['id'],passed=checked['passed']))
                atomic_json(path,saved)
                print(f"{label}/retention: {len(saved['retention'])}/15",flush=True)
    saved['completed']=True; atomic_json(path,saved)


def comparison(folder, candidate, suite):
    from qwen35_server import validate_adapter
    from qwen35_train import MODEL
    config_path=folder/'evaluation.json'
    before_path=folder/'baseline'/(suite+'.json'); after_path=folder/candidate/(suite+'.json')
    adapter=folder/candidate/'adapter'
    try:
        config=read(config_path); expected=config[suite]
        wanted=[(e['condition'],e['task']['id']) for e in expected]
        validate_adapter(adapter,MODEL,config['model_revision'])
        weight_hash=sha256(adapter/'adapter_model.safetensors')
        before=read(before_path); after=read(after_path)
        common=dict(evaluation_sha256=sha256(config_path),suite=suite)
        baseline_hash=config.get('baseline_adapter_sha256')
        if baseline_hash is not None and (not isinstance(baseline_hash,str) or len(baseline_hash)!=64
                or any(c not in '0123456789abcdef' for c in baseline_hash)):
            raise ValueError('Invalid frozen baseline adapter digest')
        if (not isinstance(before,dict) or not isinstance(after,dict)
                or before.get('binding')!=dict(common,adapter_sha256=baseline_hash)
                or after.get('binding')!=dict(common,adapter_sha256=weight_hash)):
            raise ValueError('Comparison requires the original base and this completed candidate adapter')
    except (KeyError,TypeError,AttributeError,FileNotFoundError) as error:
        raise ValueError('Comparison requires intact evaluation and adapter evidence') from error
    for report in (before,after):
        rows=report.get('rows')
        if (report.get('completed') is not True or not isinstance(rows,list)
                or any(not isinstance(r,dict) or type(r.get('passed')) is not bool
                       or type(r.get('recursive_passed')) is not bool or r.get('infrastructure_errors')
                       or r['recursive_passed'] and not r['passed'] for r in rows)
                or [(r.get('condition'),r.get('id')) for r in rows]!=wanted):
            raise ValueError('Comparison requires complete identical tasks and settings')
    gained=[e['task']['repository'] for e,b,a in zip(expected,before['rows'],after['rows']) if a['passed'] and not b['passed']]
    lost=[a['id'] for b,a in zip(before['rows'],after['rows']) if b['passed'] and not a['passed']]
    recursive=[a['id'] for b,a in zip(before['rows'],after['rows']) if a['condition']=='recursive'
               and a['recursive_passed'] and not b['recursive_passed']]
    lost_recursive=[a['id'] for b,a in zip(before['rows'],after['rows']) if b['recursive_passed'] and not a['recursive_passed']]
    retention_lost=[]
    if suite=='development':
        ids=[t['id'] for t in config['retention']]
        for report in (before,after):
            rows=report.get('retention')
            if (not isinstance(rows,list) or any(not isinstance(r,dict) or type(r.get('passed')) is not bool for r in rows)
                    or [r.get('id') for r in rows]!=ids):
                raise ValueError('Retention results are incomplete')
        retention_lost=[a['id'] for b,a in zip(before['retention'],after['retention']) if b['passed'] and not a['passed']]
    accepted=not(lost or lost_recursive or retention_lost) and len(set(gained)) >= (2 if suite=='development' else 1)
    if suite=='development':accepted=accepted and bool(recursive)
    result=dict(completed=True,pilot_passed=accepted,full_loop_authorized=False,
        before=sum(r['passed'] for r in before['rows']),after=sum(r['passed'] for r in after['rows']),
        total=len(expected),gained_families=sorted(set(gained)),lost=lost,recursive_gains=recursive,
        recursive_losses=lost_recursive,retention_losses=retention_lost,
        inputs={str(path.relative_to(folder)):sha256(path) for path in
                (config_path,before_path,after_path,adapter/'adapter_model.safetensors',
                 adapter/'adapter_config.json',adapter/'training.json')},
        reason='Full matched general-code audit still required' if accepted else 'Pilot gain requirement not met')
    atomic_json(folder/candidate/(suite+'-gate.json'),result)
    return result


def verified_curriculum(dataset, revision):
    """Require the measured complete-trajectory preflight for these exact files."""
    from qwen35_train import MODEL
    manifest_path=Path(str(dataset)+'.manifest.json')
    try:
        manifest=read(manifest_path); evidence=read(dataset.parent/'feasibility.json')
        if (any(evidence.get(key) is not True for key in
                ('passed','all_agent_actions_included','complete_trajectories'))
                or evidence.get('tokenizer_model')!=MODEL or evidence.get('tokenizer_revision')!=revision
                or evidence.get('dataset_sha256')!=sha256(dataset)
                or manifest.get('dataset_sha256')!=evidence['dataset_sha256']
                or evidence.get('manifest_sha256')!=sha256(manifest_path)
                or type(manifest.get('rows')) is not int or manifest['rows']<1):
            raise ValueError('Training requires an intact, passing complete-trajectory preflight')
    except (OSError,KeyError,TypeError,AttributeError) as error:
        raise ValueError('Training requires an intact, passing complete-trajectory preflight') from error
    return manifest['rows']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','evaluate','task','compare','train'))
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--retention',type=Path)
    parser.add_argument('--label',default='baseline')
    parser.add_argument('--adapter',type=Path)
    parser.add_argument('--recipe',type=int,choices=(0,1),default=0)
    parser.add_argument('--suite',choices=('development','holdout'),default='development')
    args=parser.parse_args()
    if args.action=='freeze':freeze(args.folder,args.retention)
    elif args.action=='task':atomic_json(args.folder/'task-result.json',evaluate_one(read(args.folder/'task.json')))
    elif args.action=='compare':print(json.dumps(comparison(args.folder,args.label,args.suite)))
    else:
        import fcntl
        def interrupt(*unused):raise KeyboardInterrupt('Finite pilot interrupted')
        signal.signal(signal.SIGTERM,interrupt)
        with (ROOT/'.cache/learning/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for port in (8080,8090):
                with socket.socket() as probe:
                    if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('GPU server already running')
            if args.action=='train':
                config=read(args.folder/'evaluation.json')
                if config['sources']!=source_hashes():raise ValueError('Pilot source changed')
                dataset=args.folder/'curriculum/train.jsonl'
                minimum=verified_curriculum(dataset,config['model_revision'])
                output=args.folder/args.label/'adapter'
                command=[str(ROOT/'.cache/qwen35-env/bin/python'),str(ROOT/'qwen35_train.py'),
                    '--dataset',str(dataset),'--manifest',str(dataset)+'.manifest.json',
                    '--output',str(output),'--revision',config['model_revision'],
                    '--min-examples',str(minimum),'--min-per-language','1','--languages',
                    'python','go','typescript','rust','dart','--save-steps','25']
                for key,value in config['recipes'][args.recipe].items():
                    command += ['--'+key.replace('_','-'),str(value)]
                if (output/'request.json').exists():command+=['--resume']
                subprocess.run(command,check=True)
            else:evaluate(args.folder,args.label,args.adapter,args.suite)


if __name__=='__main__':main()
