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
from urllib.error import HTTPError

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


def task_messages(task):
    if task.get('kind') == 'teacher_function':
        from teacher_data import messages as teacher_messages
        return teacher_messages(task)
    return messages(task)


def check_task(task, answer):
    if task.get('kind') == 'teacher_function':
        from teacher_data import check
        return check(task, answer)
    return check_program(task['language'], answer, task['cases'])


def prepare(session, folder, config):
    if shutil.disk_usage(session).free < 10*1024**3:
        raise RuntimeError('Less than 10 GiB free disk; free space before more training')
    pool = session/'pool'; pool.mkdir(exist_ok=True)
    target = folder/'tasks.json'
    if target.exists():
        saved = read(target); matched_tasks(saved['train']+saved['validation'])
        value = dict(completed=True, tasks_sha256=sha256(target))
        if (folder/'teacher.json').exists(): value['teacher_sha256'] = sha256(folder/'teacher.json')
        return value
    teacher = None
    if config.get('teacher_examples', 0) and int(folder.name.split('-')[-1]) >= config.get('teacher_start_cycle',1):
        from teacher_data import prepare_round
        progress(folder, 'verify-teacher', task='Loading pinned NVIDIA teacher examples; then checking them locally')
        teacher = prepare_round(session,folder,config['teacher_examples'],config.get('teacher_validation',20),progress)
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
    if teacher:
        from teacher_data import task
        validation += [task(entry,'dev') for entry in teacher['validation']]
    matched_tasks(train+validation)
    assert not {t['repository'] for t in train} & {t['repository'] for t in validation}
    state = read(session/'status.json')
    atomic_json(target, dict(train=train, validation=validation, parent=state.get('retained_adapter'),
                            fixed_ids=[t['id'] for t in fixed]))
    value = dict(completed=True, tasks_sha256=sha256(target))
    if teacher: value['teacher_sha256'] = sha256(folder/'teacher.json')
    return value


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
    try:
        responses = request('http://127.0.0.1:8090', '/batch',
            {'conversations':[conversation], 'max_tokens':2048, 'temperature':temperature, 'seed':seed},
            timeout=600)['responses']
    except HTTPError as error:
        detail=error.read(4096).decode('utf-8',errors='replace')
        error_type=ValueError if 400<=error.code<500 and error.code not in (408,429) else RuntimeError
        raise error_type(f'Coding request rejected (HTTP {error.code}): {detail}') from error
    answer=responses[0]['choices'][0]['message'].get('content') or ''
    try:
        return source(answer)
    except ValueError:
        # Empty or incorrectly formatted model output is a failed answer, not a worker fault.
        return ''


def collect(folder, tasks, parent, phase, repair_turns=1):
    if type(repair_turns) is not int or not 1 <= repair_turns <= 3:
        raise ValueError('Use one to three public-feedback repair attempts')
    path = folder/(phase+'-rows.json')
    binding = dict(tasks_sha256=sha256(folder/'tasks.json'),
        adapter_sha256=sha256(Path(parent)/'adapter_model.safetensors') if parent else None)
    if repair_turns > 1: binding['repair_turns'] = repair_turns
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
            conversation = task_messages(task)
            if 'answer' not in attempt:
                attempt['answer'] = ask(conversation); atomic_json(attempt_file,attempt)
            answer = attempt['answer']
            checked = check_task(task, answer)
            first_passed = checked['passed']
            repair_messages = None
            repair_attempts = 0
            # Only public example feedback is visible to the model, never hidden test results.
            if phase == 'collect' and not checked['passed']:
                for index in range(repair_turns):
                    if checked['passed']: break
                    visible = check_program(task['language'], answer, visible_cases(task))
                    if visible['passed']: break
                    # Keep the task and latest failure; accumulating full programs overflows context.
                    correction = conversation+[{'role':'assistant','content':answer or '(empty answer)'}, {'role':'user',
                        'content':'The public checks failed:\n'+visible['feedback']+'\nReturn the complete corrected program.'}]
                    key = 'correction' if index == 0 else f'correction-{index}'
                    if key not in attempt:
                        attempt[key] = ask(correction); atomic_json(attempt_file,attempt)
                    answer = attempt[key]
                    checked = check_task(task, answer)
                    repair_attempts += 1
                    if checked['passed']:
                        repair_messages = correction+[{'role':'assistant','content':answer}]
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
                answer=answer, initial_answer=attempt['answer'], attempts=attempt, first_passed=first_passed,
                feedback=checked.get('feedback',''), corrected=bool(repair_messages), repair_attempts=repair_attempts,
                **({'repair_messages':repair_messages} if repair_messages else {}),
                sampled=sum(k.startswith('sample-') for k in attempt)))
            atomic_json(path,saved)
    return dict(completed=True, rows=saved['rows'], binding=binding)


def replay_runtime():
    from polyglot_runtime import image
    return dict(image=image(), sources={name:sha256(ROOT/name)
                for name in ('polyglot_runtime.py','evaluation.py')})


def verify_replay(session, task, answer):
    binding=dict(language=task['language'],cases=task['cases'],answer=answer,runtime=replay_runtime())
    key=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    path=session/'replay-checks'/(key+'.json')
    cached=read(path,{})
    if cached.get('binding_sha256')==key and cached.get('passed') is True:
        return True
    result=check_program(task['language'],answer,task['cases'])
    if result['passed']:
        atomic_json(path,dict(binding_sha256=key,passed=True))
    return result['passed']


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
    # An optional, immutable seed contains earlier training answers and their source tasks.
    # Its entries receive the same split, deduplication, execution and length checks as new answers.
    seed = config.get('replay_seed')
    if seed:
        seed_path = session/seed['file']
        if sha256(seed_path) != seed['sha256']:
            raise ValueError('Verified replay seed changed')
        for item in read(seed_path):
            task,value=item['task'],item['answer']
            identity=task['provenance']['row_id']
            if (task['split']!='train' or int(identity[:2],16)%5==0
                    or task['repository']!='codecontests-'+identity
                    or task['id']!='codecontests-'+identity[:24]+'-'+task['language']):
                raise ValueError('Replay seed must contain canonical training families only')
            if value['id']!=task['id'] or value['passed'] is not True:
                raise ValueError('Replay seed answer identity or verification differs')
            if task['id'] not in seen:
                seen.add(task['id'])
                candidates[task['language']].insert(0,(task,value))
    rows=[]; lesson_jobs=[]
    execution = config.get('learning_method') == 'execution_repair'
    for language in LANGUAGES:
        available=[]
        for task,value in candidates[language][-config['replay_per_language']:]:
            progress(folder,'verify-replay',len(rows),task=task['id'],language=language)
            if not verify_replay(session,task,value['answer']):
                raise ValueError('A previously verified training answer no longer passes')
            row=dict(task_id=task['id'], language=language, split='train',
                messages=messages(task)+[{'role':'assistant','content':value['answer']}],
                verification={'kind':'local_execution','passed':True},
                provenance={'dataset':task['provenance']['dataset'],'revision':task['provenance']['revision'],
                    'source_row_sha256':hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()})
            if encode_rows(tokenizer,[row],4096)[0]:
                rows.append(dict(row,example_kind='complete_code'))
                available.append(dict(task=task,answer=value))
            correction=value.get('repair_messages')
            if execution and correction:
                if correction[:2]!=messages(task) or correction[-1]!=dict(role='assistant',content=value['answer']):
                    raise ValueError('Correction context does not match the verified task and answer')
                corrected=dict(row,task_id=task['id']+':correction',source_task_id=task['id'],
                               example_kind='correction',messages=correction)
                if encode_rows(tokenizer,[corrected],4096)[0]:rows.append(corrected)
        if execution:
            # Rotate the checked demonstration tasks while preserving the complete replay pool.
            cycle=folder.name
            available.sort(key=lambda job:hashlib.sha256((cycle+job['task']['id']).encode()).hexdigest())
            lesson_jobs.extend(available[:config.get('repl_lessons_per_language',2)])
    if execution:
        from repl_learning import export_lessons
        progress(folder,'verify-repl',0,len(lesson_jobs),task='Executing inspect, fail, repair and retest lessons')
        rows.extend(r for r in export_lessons(folder,lesson_jobs) if encode_rows(tokenizer,[r],4096)[0])
    from teacher_data import training_rows
    teachers = training_rows(folder)
    teachers = [r for r in teachers if encode_rows(tokenizer,[r],4096)[0]]
    rows.extend(dict(r,example_kind='teacher_reference') for r in teachers)
    # Interleave bounded language pools without discarding useful majority-language rows.
    grouped={l:[r for r in rows if r['language']==l] for l in LANGUAGES}
    rows=[grouped[l][i] for i in range(max(map(len,grouped.values()))) for l in LANGUAGES if i<len(grouped[l])]
    if execution:
        from repl_learning import mix_rows
        rows=mix_rows(rows)
    data=folder/'train.jsonl'
    data.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    counts=dict(Counter(r['language'] for r in rows))
    kinds=Counter(r.get('example_kind','complete_code') for r in rows)
    atomic_json(Path(str(data)+'.manifest.json'), dict(schema_version=1,split='train',verification='local_execution',
        rows=len(rows),languages=counts,dataset_sha256=sha256(data),
        teacher_rows=kinds['teacher_reference'],example_kinds=dict(kinds),
        sources=([dict(dataset='nvidia/OpenCodeInstruct',license='CC-BY-4.0',
                       attribution='NVIDIA, OpenCodeInstruct (Ahmad et al., 2025)')] if teachers else []),
        scope='Locally checked reference functions and model-generated programs; public source overlap is possible.'))
    ready = len(rows)>=config.get('min_examples',100) and all(counts.get(l,0)>=config.get('min_per_language',8) for l in LANGUAGES)
    if (folder/'teacher.json').exists(): ready = ready and len(teachers)>=config['teacher_examples']//2
    if execution: ready = ready and kinds['executed_repl_repair']>0
    return dict(completed=True,ready=ready,rows=len(rows),teacher_rows=kinds['teacher_reference'],
                correction_rows=kinds['correction'],repl_rows=kinds['executed_repl_repair'],
                reason='Checked code, corrections and REPL lessons ready' if ready else 'More checked examples across all languages or valid REPL lessons needed',
                languages=counts,data_sha256=sha256(data))


def training_recipe(session, folder, config):
    # A resumed optimizer must keep the recipe it started with.
    saved=read(folder/'adapter/request.json')
    if saved:
        return {key:saved[key] for key in ('max_steps','max_epochs','learning_rate')}
    recipe=dict(max_steps=config['steps_per_round'],max_epochs=config.get('max_epochs',1),
                learning_rate=config['learning_rate'])
    teacher = (folder/'teacher.json').exists()
    base_rate = config.get('teacher_learning_rate',2e-5) if teacher else config['learning_rate']
    if teacher and config.get('learning_method') != 'execution_repair':
        recipe.update(method='teacher_mix', learning_rate=base_rate, max_steps=256)
    elif config.get('learning_method') == 'execution_repair':
        recipe['method']='execution_repair'
    for path in sorted(session.glob('round-*/finish.json'),reverse=True):
        if path.parent.name>=folder.name: continue
        if teacher != (path.parent/'teacher.json').exists(): continue
        report=read(path)
        comparisons=report.get('comparisons',{}).get('training_effect',{})
        suite='audit' if comparisons.get('audit') else 'validation'
        audit=comparisons.get(suite) if teacher or suite=='audit' else None
        if not report.get('completed') or not audit: continue
        run=path.parent/'adapter/run.json'
        if report.get('inputs',{}).get(str(run.relative_to(session)))!=sha256(run):
            raise ValueError('Completed training recipe evidence changed')
        rate=min(base_rate,read(run)['learning_rate'])
        lost=sum(len(row['lost']) for row in audit.values())
        if lost: rate=max(min(rate,base_rate/4),rate/2)
        recipe.update(learning_rate=rate,previous_audit=str(path.relative_to(session)),
                      previous_audit_sha256=sha256(path),previous_lost=lost,previous_evidence_suite=suite)
        break
    return recipe


def train_block(session, folder, args, config, parent):
    if not read(folder/'export.json')['ready']:
        return dict(status='completed',skipped=True,reason='Collecting more verified replay before training')
    if sha256(folder/'train.jsonl') != read(folder/'export.json')['data_sha256']:
        raise ValueError('Verified training data changed after export')
    recipe=training_recipe(session,folder,config)
    if not (folder/'recipe.json').exists(): atomic_json(folder/'recipe.json',recipe)
    print(f"Training recipe: rate={recipe['learning_rate']:g}, epochs={recipe['max_epochs']}, "
          f"maximum steps={recipe['max_steps']}",flush=True)
    import qwen35_train
    command=['--dataset',str(folder/'train.jsonl'),'--manifest',str(folder/'train.jsonl.manifest.json'),
        '--output',str(folder/'adapter'),'--revision',REVISION,'--max-steps',str(recipe['max_steps']),
        '--max-epochs',str(recipe['max_epochs']),'--learning-rate',str(recipe['learning_rate']),'--max-length','4096',
        '--rank','8','--gradient-accumulation','8','--save-steps','5','--min-examples',str(config.get('min_examples',20)),
        '--min-per-language',str(config.get('min_per_language',1)),'--languages',*LANGUAGES,'--pause-file',str(args.pause_file)]
    if parent: command += ['--initial-adapter',parent]
    if args.resume: command += ['--resume']
    sys.argv=['qwen35_train.py',*command]
    qwen35_train.main()
    return None


def audit_due(session, folder, gate, config):
    if gate.get('skipped'):
        return False
    if gate['promising']:
        return True
    # Once started, both halves remain resumable even across a recipe migration.
    if (folder/'audit-parent/binding.json').exists():
        return True
    if (folder/'teacher.json').exists():
        since_audit=0
        for result in sorted(session.glob('round-*/adapter/training.json'),reverse=True):
            prior=result.parent.parent
            if prior.name>folder.name or not (prior/'teacher.json').exists(): continue
            report=read(prior/'finish.json',{})
            if prior.name<folder.name and report.get('completed') and report.get('comparisons',{}).get('training_effect',{}).get('audit'):
                break
            training=read(result)
            since_audit+=training.get('status')=='completed' and not training.get('skipped')
        return since_audit>=config.get('teacher_audit_every_updates',3)
    trained=sum(not read(p).get('skipped') and read(p).get('status')=='completed'
                for p in session.glob('round-*/adapter/training.json') if p.parent.parent.name<=folder.name)
    interval=config.get('audit_every_updates',20)
    return trained>0 and trained%interval==0


def validation_gate(folder, config):
    candidate = read(folder/'candidate.json')
    if candidate.get('skipped'):
        return dict(completed=True, skipped=True)
    direct = promotion(read(folder/'baseline.json')['rows'], candidate['rows'])
    if config.get('learning_method') != 'execution_repair':
        return dict(completed=True, **direct)
    before, after = read(folder/'repl-baseline.json'), read(folder/'repl-candidate.json')
    if not before or not after or before.get('skipped') or after.get('skipped'):
        raise ValueError('Execution learning requires a complete matched REPL comparison')
    repl = promotion(before['rows'], after['rows'])
    return dict(completed=True, **{**direct, 'promising':bool(direct['gained'] or repl['gained'])
                and not direct['lost'] and not repl['lost']}, repl=repl)


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
        if 'repl' in gate:
            comparisons['repl']=compare(by_lang(read(folder/'repl-baseline.json')['rows']),
                                        by_lang(read(folder/'repl-candidate.json')['rows']))
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
    reasons=[]
    for suite, languages in comparisons.items():
        lost=sum(len(r['lost']) for r in languages.values())
        gained=sum(len(r['gained']) for r in languages.values())
        if lost:reasons.append(f'{suite}: {lost} previously passing tasks lost ({gained} gained)')
    if not gate.get('skipped') and not gate['promising'] and not reasons:
        reasons.append('No new validation or REPL pass without regressions')
    if gate.get('skipped'):reasons.append(read(folder/'export.json',{}).get('reason','More verified examples needed'))
    files=[p for p in folder.rglob('*.json') if p.name!='finish.json' and 'checkpoint-' not in str(p)]
    result=dict(completed=True,accepted=accepted,retained_adapter=retained,decision=decision,reasons=reasons,
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
        if phase=='collect': value=collect(folder,info['train'],parent,phase,config.get('repair_turns',1))
        elif phase=='export': value=export_data(session,folder,config)
        elif phase=='baseline': value=collect(folder,info['validation'],parent,phase) if read(folder/'export.json')['ready'] else dict(completed=True,skipped=True)
        elif phase=='train':
            value=train_block(session,folder,args,config,parent)
            if value: atomic_json(folder/'adapter/training.json',value)
            return
        elif phase=='candidate': value=collect(folder,info['validation'],str(folder/'adapter'),phase) if not read(folder/'adapter/training.json').get('skipped') else dict(completed=True,skipped=True)
        elif phase in ('repl-baseline','repl-candidate'):
            from repl_learning import evaluate
            value=evaluate(folder,info['validation'],parent if phase=='repl-baseline' else str(folder/'adapter'),phase,config) if read(folder/'export.json')['ready'] else dict(completed=True,skipped=True)
        elif phase=='gate':
            value=validation_gate(folder,config)
        elif phase.startswith('audit-'):
            gate=read(folder/'gate.json')
            value=audit(folder,parent,str(folder/'adapter'),phase,audit_due(session,folder,gate,config))
        elif phase=='finish': value=finish(session,folder,parent)
        else: raise ValueError('Unknown coding loop phase')
    atomic_json(folder/(phase+'.json'),value)
    progress(folder,phase,1,1)


if __name__=='__main__':
    try: main()
    except ValueError as error:
        print(f'STAGE_INVALID: {error}',file=sys.stderr)
        sys.exit(2)
