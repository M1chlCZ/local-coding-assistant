"""Resumable stages for verified complete-code learning. All artifacts stay private."""
import argparse
from collections import Counter
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from coding_loop import promotion
from focused_experiment import ROOT, read
from qwen35_train import atomic_json, sha256
from polyglot_runtime import LANGUAGES, check_program
from code_recipe import messages, source, visible_cases, balanced_order
from launcher import health

REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def progress(folder, phase, done=0, total=0, **extra):
    atomic_json(folder/'progress.json', dict(phase=phase, completed=done, total=total,
        heartbeat_at=time.time(), **extra))
    print(f'{phase}: {done}/{total} '+extra.get('task',''), flush=True)


def matched_tasks(tasks):
    ids = [t['id'] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate curriculum task identity')
    assignments = {}
    for t in tasks:
        previous = assignments.setdefault(t['repository'], t['split'])
        if previous != t['split']:
            raise ValueError('A problem family crosses training and evaluation splits')


def prepare(session, folder, config):
    if shutil.disk_usage(session).free < 10*1024**3:
        raise RuntimeError('Less than 10 GiB free disk; free space before more training')
    pool = session/'pool'; pool.mkdir(exist_ok=True)
    target = folder/'tasks.json'
    if target.exists():
        saved = read(target); matched_tasks(saved['train']+saved['validation'])
        return dict(completed=True, tasks_sha256=sha256(target))
    used = set()
    for path in session.glob('round-*/tasks.json'):
        value = read(path)
        used.update(t['id'] for t in value['train']+value['validation'])
    fixed = read(session/'development.json', [])
    used.update(t['id'] for t in fixed)
    while True:
        tasks = []
        from polyglot_data import MANIFEST, prepare_batch
        for path in sorted(pool.glob('batch-*/tasks.json')):
            manifest = read(path.parent/'manifest.json')
            if manifest['tasks_sha256'] != sha256(path) or manifest['source_manifest_sha256'] != sha256(MANIFEST):
                raise ValueError('Curriculum source integrity check failed')
            tasks.extend(t for t in read(path) if t['id'] not in used)
        train = balanced_order([t for t in tasks if t['split']=='train'], config['fresh_tasks'])
        dev = balanced_order([t for t in tasks if t['split']=='dev'])
        needed = config['validation_tasks']*(1 if fixed else 2)
        if len(train) >= config['fresh_tasks'] and len(dev) >= needed:
            break
        progress(folder, 'validate-source', len(tasks), task='Checking fresh source problems in isolated containers')
        prepare_batch(pool)
    if not fixed:
        fixed = dev[:config['validation_tasks']]
        atomic_json(session/'development.json', fixed)
        dev = dev[config['validation_tasks']:]
    validation = fixed+dev[:config['validation_tasks']]
    matched_tasks(train+validation)
    assert not {t['repository'] for t in train} & {t['repository'] for t in validation}
    state = read(session/'status.json')
    atomic_json(target, dict(train=train, validation=validation, parent=state.get('retained_adapter'),
                            fixed_ids=[t['id'] for t in fixed]))
    return dict(completed=True, tasks_sha256=sha256(target))


@contextlib.contextmanager
def server(adapter, folder):
    from evaluation import request
    command = [str(ROOT/'.cache/qwen35-env/bin/python'), str(ROOT/'qwen35_server.py'),
               '--revision', REVISION, '--batch-size','2','--output-limit','2048']
    if adapter:
        command += ['--adapter',str(adapter)]
    process = None
    try:
        with (folder/'server.log').open('ab') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        deadline = time.monotonic()+900
        while not health(8090):
            if process.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('Coding model failed to load; inspect server.log')
            progress(folder, 'load-model')
            time.sleep(3)
        request('http://127.0.0.1:8090', '/mode', {'mode':'adapter' if adapter else 'base'}, timeout=30)
        yield
    finally:
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait()


def ask(conversation, temperature=0, seed=3407):
    from evaluation import request
    responses = request('http://127.0.0.1:8090', '/batch',
        {'conversations':[conversation], 'max_tokens':2048, 'temperature':temperature, 'seed':seed},
        timeout=600)['responses']
    answer=responses[0]['choices'][0]['message'].get('content') or ''
    try:
        return source(answer)
    except ValueError:
        # Empty or incorrectly formatted model output is a failed answer, not a worker fault.
        return ''


def collect(folder, tasks, parent, phase):
    path = folder/(phase+'-rows.json')
    binding = dict(tasks_sha256=sha256(folder/'tasks.json'),
        adapter_sha256=sha256(Path(parent)/'adapter_model.safetensors') if parent else None)
    saved = read(path, dict(binding=binding, rows=[]))
    if saved['binding'] != binding or [r['id'] for r in saved['rows']] != [t['id'] for t in tasks[:len(saved['rows'])]]:
        raise ValueError('Saved model answers do not match the frozen tasks and weights')
    if len(saved['rows']) == len(tasks):
        return dict(completed=True, rows=saved['rows'], binding=binding)
    with server(parent, folder):
        for task in tasks[len(saved['rows']):]:
            progress(folder, phase, len(saved['rows']), len(tasks), task=task['id'], language=task['language'])
            attempt_file = folder/(phase+'-attempt.json')
            attempt = read(attempt_file, {})
            if attempt.get('id') != task['id']:
                attempt = {'id':task['id']}
            conversation = messages(task)
            if 'answer' not in attempt:
                attempt['answer'] = ask(conversation); atomic_json(attempt_file,attempt)
            answer = attempt['answer']
            checked = check_program(task['language'], answer, task['cases'])
            # Only public example feedback is visible to the model, never hidden test results.
            if phase == 'collect' and not checked['passed']:
                visible = check_program(task['language'], answer, visible_cases(task))
                if not visible['passed']:
                    if 'correction' not in attempt:
                        correction = conversation+[{'role':'assistant','content':answer}, {'role':'user',
                            'content':'The public checks failed:\n'+visible['feedback']+'\nReturn the complete corrected program.'}]
                        attempt['correction'] = ask(correction); atomic_json(attempt_file,attempt)
                    answer = attempt['correction']
                    checked = check_program(task['language'], answer, task['cases'])
                # Execution-filtered self-training: bounded independent samples, never test answers.
                for index in range(2):
                    if checked['passed']: break
                    key = f'sample-{index}'
                    seed = int(hashlib.sha256((task['id']+key).encode()).hexdigest()[:8],16)
                    if key not in attempt:
                        attempt[key] = ask(conversation, temperature=.7, seed=seed)
                        atomic_json(attempt_file,attempt)
                    answer = attempt[key]
                    checked = check_program(task['language'],answer,task['cases'])
            saved['rows'].append(dict(id=task['id'], language=task['language'], passed=checked['passed'],
                answer=answer, feedback=checked.get('feedback',''), corrected='correction' in attempt,
                sampled=sum(k.startswith('sample-') for k in attempt)))
            atomic_json(path,saved)
    return dict(completed=True, rows=saved['rows'], binding=binding)


def export_data(session, folder, config):
    from transformers import AutoTokenizer
    from qwen35_train import encode_rows, MODEL
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    candidates = {l:[] for l in LANGUAGES}
    seen = set()
    # Replay only privately verified training rows, never validation answers.
    for path in sorted(session.glob('round-*/collect.json')):
        if path.parent.name > folder.name: continue
        if sha256(path.parent/'tasks.json') != read(path.parent/'prepare.json')['tasks_sha256']:
            raise ValueError('Replay task registry changed')
        info = read(path.parent/'tasks.json'); known={t['id']:t for t in info['train']}
        for value in read(path)['rows']:
            task=known[value['id']]
            if task['split']!='train' or not value['passed'] or task['id'] in seen: continue
            seen.add(task['id'])
            candidates[task['language']].append((task,value))
    rows=[]
    for language in LANGUAGES:
        for task,value in candidates[language][-config['replay_per_language']:]:
            progress(folder,'verify-replay',len(rows),task=task['id'],language=language)
            if not check_program(language,value['answer'],task['cases'])['passed']:
                raise ValueError('A previously verified training answer no longer passes')
            row=dict(task_id=task['id'], language=language, split='train',
                messages=messages(task)+[{'role':'assistant','content':value['answer']}],
                verification={'kind':'local_execution','passed':True},
                provenance={'dataset':task['provenance']['dataset'],'revision':task['provenance']['revision'],
                    'source_row_sha256':hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()})
            if encode_rows(tokenizer,[row],4096)[0]: rows.append(row)
    # Equal language coverage prevents easier Python targets from dominating the update.
    grouped={l:[r for r in rows if r['language']==l] for l in LANGUAGES}
    per_language=min(map(len,grouped.values()))
    rows=[grouped[l][i] for i in range(per_language) for l in LANGUAGES]
    data=folder/'train.jsonl'
    data.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    counts=dict(Counter(r['language'] for r in rows))
    atomic_json(Path(str(data)+'.manifest.json'), dict(schema_version=1,split='train',verification='local_execution',
        rows=len(rows),languages=counts,dataset_sha256=sha256(data),
        scope='Complete programs locally checked against independent source tests; public source overlap is possible.'))
    ready = len(rows)>=20 and all(counts.get(l,0)>=1 for l in LANGUAGES)
    return dict(completed=True,ready=ready,rows=len(rows),languages=counts,data_sha256=sha256(data))


def train_block(session, folder, args, config, parent):
    if not read(folder/'export.json')['ready']:
        return dict(status='completed',skipped=True,reason='Collecting more verified replay before training')
    if sha256(folder/'train.jsonl') != read(folder/'export.json')['data_sha256']:
        raise ValueError('Verified training data changed after export')
    import qwen35_train
    command=['--dataset',str(folder/'train.jsonl'),'--manifest',str(folder/'train.jsonl.manifest.json'),
        '--output',str(folder/'adapter'),'--revision',REVISION,'--max-steps',str(config['steps_per_round']),
        '--max-epochs','1','--learning-rate',str(config['learning_rate']),'--max-length','4096',
        '--rank','8','--gradient-accumulation','8','--save-steps','5','--min-examples','20',
        '--min-per-language','1','--languages',*LANGUAGES,'--pause-file',str(args.pause_file)]
    if parent: command += ['--initial-adapter',parent]
    if args.resume: command += ['--resume']
    sys.argv=['qwen35_train.py',*command]
    qwen35_train.main()
    return None


def audit(folder, parent, candidate, phase, due):
    if not due: return dict(completed=True,skipped=True)
    from focused_evaluation import run
    target=parent if phase=='audit-parent' else candidate
    output=folder/phase
    args=argparse.Namespace(engine='qwen35',mode='adapter' if target else 'base',adapter=Path(target) if target else None,
        suite='audit',batch_size=2,smoke_limit=None,output=output,selection=None,repairs=None,replay_data=None)
    run(args)
    return dict(completed=True,summary_sha256=sha256(output/'summary.json'))


def finish(session, folder, parent):
    from focused_report import compare
    gate=read(folder/'gate.json'); accepted=False
    comparisons={}; models={}
    if not gate.get('skipped'):
        before=read(folder/'baseline.json')['rows']; after=read(folder/'candidate.json')['rows']
        by_lang=lambda rows: {l:[r for r in rows if r['language']==l] for l in LANGUAGES}
        comparisons['validation']=compare(by_lang(before),by_lang(after))
        if not read(folder/'audit-parent.json').get('skipped'):
            full=[]; bindings=[]
            for name in ('audit-parent','audit-candidate'):
                summary=read(folder/name/'summary.json')
                if not summary.get('completed') or summary['total']!=790:
                    raise ValueError('Promotion requires all 790 benchmark tasks')
                bindings.append(summary['binding'])
                rows={l:read(folder/name/(l+'.json')) for l in LANGUAGES}
                checked=compare(rows,rows)
                if (sum(v['total'] for v in checked.values())!=790
                        or sum(v['after'] for v in checked.values())!=summary['passed']
                        or any(checked[l]['total']!=summary['languages'][l]['total']
                               or checked[l]['after']!=summary['languages'][l]['passed'] for l in LANGUAGES)):
                    raise ValueError('Benchmark summary differs from complete functional rows')
                full.append(rows)
                models['retained' if name=='audit-parent' else 'candidate']={'audit':{k:summary[k] for k in ('passed','total','languages','macro_pass_at_1')}}
            keys=('datasets','max_output_tokens','temperature','thinking','batch_size','prompt','container_image','sources','engine','model','revision','precision','runtime')
            if any(bindings[0].get(k)!=bindings[1].get(k) for k in keys):
                raise ValueError('Full audit conditions differ')
            comparisons['audit']=compare(*full)
            before_all=[r for l in LANGUAGES for r in full[0][l]]
            after_all=[r for l in LANGUAGES for r in full[1][l]]
            result=promotion(before_all,after_all)
            accepted=gate['promising'] and not result['lost'] and sum(r['passed'] for r in after_all)>=sum(r['passed'] for r in before_all)
    retained=str(folder/'adapter') if accepted else parent
    decision=('Candidate accepted after validation and full audit' if accepted else
              'Candidate rejected; retained model unchanged' if not gate.get('skipped') else
              'More verified examples needed; continuing collection')
    files=[p for p in folder.rglob('*.json') if p.name!='finish.json' and 'checkpoint-' not in str(p)]
    result=dict(completed=True,accepted=accepted,retained_adapter=retained,decision=decision,
        comparisons={'training_effect':comparisons},models=models,
        inputs={str(p.relative_to(session)):sha256(p) for p in files if p.name not in ('progress.json','request.json')},
        limitations=['Validation and public function tests are reused development checks, not proof of general coding quality.',
                    'Fresh validation problem families are reserved before training and never enter replay.'])
    # Keep compact reports; discard only this rejected candidate's replaceable weights.
    if not accepted and (folder/'adapter').exists():
        for child in (folder/'adapter').glob('checkpoint-*'):
            if child.is_dir(): shutil.rmtree(child)
        for name in ('adapter_model.safetensors',): (folder/'adapter'/name).unlink(missing_ok=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True)
    p.add_argument('--round',type=int,required=True);p.add_argument('--phase',required=True)
    p.add_argument('--pause-file',type=Path,required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();session=args.session.resolve();folder=session/f'round-{args.round:06d}'
    config=read(session/'config.json');phase=args.phase
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    progress(folder,phase)
    if phase=='prepare': value=prepare(session,folder,config)
    else:
        info=read(folder/'tasks.json');parent=info['parent']
        if sha256(folder/'tasks.json')!=read(folder/'prepare.json')['tasks_sha256']:
            raise ValueError('Assigned train and validation tasks changed')
        if phase=='collect': value=collect(folder,info['train'],parent,phase)
        elif phase=='export': value=export_data(session,folder,config)
        elif phase=='baseline': value=collect(folder,info['validation'],parent,phase) if read(folder/'export.json')['ready'] else dict(completed=True,skipped=True)
        elif phase=='train':
            value=train_block(session,folder,args,config,parent)
            if value: atomic_json(folder/'adapter/training.json',value)
            return
        elif phase=='candidate': value=collect(folder,info['validation'],str(folder/'adapter'),phase) if not read(folder/'adapter/training.json').get('skipped') else dict(completed=True,skipped=True)
        elif phase=='gate':
            candidate=read(folder/'candidate.json')
            value=dict(completed=True,skipped=True) if candidate.get('skipped') else dict(completed=True,**promotion(read(folder/'baseline.json')['rows'],candidate['rows']))
        elif phase.startswith('audit-'):
            gate=read(folder/'gate.json')
            value=audit(folder,parent,str(folder/'adapter'),phase,not gate.get('skipped') and (gate['promising'] or args.round%5==0))
        elif phase=='finish': value=finish(session,folder,parent)
        else: raise ValueError('Unknown coding loop phase')
    atomic_json(folder/(phase+'.json'),value)
    progress(folder,phase,1,1)


if __name__=='__main__': main()
