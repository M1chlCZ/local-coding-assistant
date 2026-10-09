"""Finite matched teacher-answer versus corrected-student-answer experiment."""
import copy,hashlib,json,time
from collections import Counter
from pathlib import Path

from code_recipe import balanced_order,export_code,solve,solve_batch
from recursive_agent import grade
from polyglot_runtime import LANGUAGES
from train_adapter import atomic_json,encode_row,verified_checkpoint
from training_data import load_verified,manifest_path,registry,sha256


def read(path):
    return json.loads(Path(path).read_text())


def recover_incomplete(path):
    companion=manifest_path(path)
    if path.exists()!=companion.exists():
        orphan=path if path.exists() else companion
        orphan.rename(orphan.with_name(orphan.name+f'.incomplete-{time.time_ns()}'))


def student_report(path,tasks_path,weights,ids):
    binding={'tasks_sha256':sha256(tasks_path),'adapter_sha256':sha256(weights),'ids':ids}
    value=read(path) if Path(path).exists() else {'binding':binding,'tasks':[]}
    if value.get('binding')!=binding or [r['id'] for r in value['tasks']]!=ids[:len(value['tasks'])]:
        raise ValueError('Student collection binding changed')
    return value


def paired_reports(tasks_path,teacher,students,corrections):
    known=registry(tasks_path)
    if (teacher.get('schema_version')!=1 or teacher.get('split')!='train'
            or teacher.get('tasks_sha256')!=sha256(tasks_path) or any(t['split']!='train' for t in known.values())):
        raise ValueError('Teaching pairs require registered training data only')
    for rows in (teacher['tasks'],students,corrections):
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate teaching answer')
        for row in rows:
            task=known.get(row['id'])
            if task is None or any(row.get(k)!=task[k] for k in ('id','repository','split')):
                raise ValueError('Teaching answer does not match training registry')
    failed={r['id'] for r in students if r.get('passed') is not True}
    replacements={r['id']:r for r in corrections if r.get('passed') is True and r.get('mode')=='direct' and r['id'] in failed}
    result=copy.deepcopy(teacher);count=0
    for index,row in enumerate(result['tasks']):
        if row.get('passed') is not True or row.get('mode')!='direct':raise ValueError('Control teacher answer is not verified')
        replacement=replacements.get(row['id'])
        if replacement and replacement.get('patch')!=row.get('patch'):
            result['tasks'][index]=copy.deepcopy(replacement);count+=1
    return result,count


def collect(session,folder):
    import learning_session as learning
    from training_feedback import collect as correct
    known=registry(folder/'tasks.json');teacher=read(folder/'teacher.json')
    paired_reports(folder/'tasks.json',teacher,[],[])
    ids={r['id'] for r in teacher['tasks']}
    tasks=balanced_order([t for t in known.values() if t['id'] in ids],limit=40)
    if len(tasks)<5:raise ValueError('Teaching trial requires all five languages')
    adapter=Path(session.state['trial_origin_adapter'])
    verified_checkpoint(adapter,read(adapter.parent/'run.json'))
    path=folder/'student.json';students=student_report(path,folder/'tasks.json',adapter/'adapter_model.safetensors',[t['id'] for t in tasks])
    if len(students['tasks'])<len(tasks):
        command=[str(learning.ROOT/'.cache/train-env/bin/python'),str(learning.ROOT/'research/adapter_eval_server.py'),'--adapter',str(adapter)]
        if not session.server(command,8090,folder/'student.log'):return
        from evaluation import request
        request('http://127.0.0.1:8090','/mode',{'mode':'adapter'})
        for offset in range(len(students['tasks']),len(tasks),4):
            session.save(detail=f'Student training attempts: {offset}/{len(tasks)}')
            if session.interrupted():return
            chunk=tasks[offset:offset+4]
            rows=solve_batch(chunk,'http://127.0.0.1:8090',**learning.TEACHER_LIMITS)
            for task,row in zip(chunk,rows):
                if row.get('patch'):row.update(grade(task,row['patch']))
            students['tasks'].extend(rows);atomic_json(path,students)
    session.stop_child()
    failed=[r for r in students['tasks'] if r.get('passed') is not True]
    path=folder/'corrections.json'
    corrections=read(path) if path.exists() else {**teacher,'tasks':[]}
    if [r['id'] for r in corrections['tasks']]!=[r['id'] for r in failed[:len(corrections['tasks'])]]:
        raise ValueError('Correction progress changed')
    if len(corrections['tasks'])<len(failed):
        command=[str(learning.ROOT/'.cache/rlm-env/bin/python'),str(learning.ROOT/'wsl_server.py'),'--model',session.state['teacher_model']]
        if not session.server(command,8080,folder/'teacher-corrections.log'):return
        for student in failed[len(corrections['tasks']):]:
            task=known[student['id']]
            def progress(number):session.save(detail=f'Teacher correcting student: {len(corrections["tasks"])}/{len(failed)}')
            row=correct(task,'http://127.0.0.1:8080',folder/'attempts'/(hashlib.sha256(task['id'].encode()).hexdigest()+'.json'),
                sha256(folder/'tasks.json'),solve=solve,grade=grade,settings=learning.quality_settings(session.state),
                limits=learning.TEACHER_LIMITS,interrupted=session.interrupted,before_attempt=progress,
                remaining_seconds=lambda:session.state['limit_seconds']-session.state['active_seconds']-(time.monotonic()-session.clock),
                initial_row=student)
            if row is None:return
            corrections['tasks'].append(row);atomic_json(path,corrections)
    atomic_json(path,corrections)
    session.save(phase='export',detail='Student corrections collected; preparing matched datasets')


def filter_pairs(folder):
    """Tokenizer-only subprocess; remove a row from both arms if either target is too long."""
    from transformers import AutoTokenizer
    from research.student_benchmark import MODEL,REVISION
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    rows={arm:load_verified(folder/f'raw-{arm}.jsonl',folder/'tasks.json') for arm in ('control','correction')}
    if [r['task_id'] for r in rows['control']]!=[r['task_id'] for r in rows['correction']]:
        raise ValueError('Paired datasets must contain the same tasks in the same order')
    ids=[a['task_id'] for a,b in zip(rows['control'],rows['correction']) if encode_row(tokenizer,a,4096) and encode_row(tokenizer,b,4096)]
    atomic_json(folder/'paired-fit.json',{'ids':ids,'inputs':{a:sha256(folder/f'raw-{a}.jsonl') for a in rows},'model':MODEL,'revision':REVISION})


def dataset(session,folder):
    import learning_session as learning
    session.stop_child()
    teacher=read(folder/'teacher.json')
    paired,count=paired_reports(folder/'tasks.json',teacher,read(folder/'student.json')['tasks'],read(folder/'corrections.json')['tasks'])
    if not count:
        session.save(status='completed',completion_reason='no_verified_corrections',detail='No distinct verified student corrections; training skipped')
        return
    def progress(done,total):
        session.save(detail=f'Checking teaching targets: {done}/{total}')
        if session.interrupted():raise InterruptedError('Teaching preparation paused')
    for arm,report in (('control',teacher),('correction',paired)):
        path=folder/f'targets-{arm}.json';atomic_json(path,report)
        recover_incomplete(folder/f'raw-{arm}.jsonl')
        if not (folder/f'raw-{arm}.jsonl').exists():
            try:export_code([path],folder/'tasks.json',folder/f'raw-{arm}.jsonl',progress=progress)
            except InterruptedError:return
    if not (folder/'paired-fit.json').exists():
        session.spawn([str(learning.ROOT/'.cache/train-env/bin/python'),str(Path(__file__)),str(folder)],folder/'paired-fit.log')
        while session.child.poll() is None:
            session.save(detail='Checking equal task coverage and token budgets')
            if session.interrupted():session.stop_child();return
            time.sleep(1)
        if session.child.returncode:raise RuntimeError('Matched token validation failed; inspect paired-fit.log')
        session.stop_child()
    fit=read(folder/'paired-fit.json');ids=set(fit['ids'])
    count=sum(a['id'] in ids and a['patch']!=b['patch'] for a,b in zip(teacher['tasks'],paired['tasks']))
    if not count:
        session.save(status='completed',completion_reason='no_verified_corrections',detail='No distinct corrections fit both token budgets; training skipped')
        return
    for arm in ('control','correction'):
        raw=folder/f'raw-{arm}.jsonl'
        if sha256(raw)!=fit['inputs'][arm]:raise ValueError('Paired dataset changed after token validation')
        rows=[r for r in load_verified(raw,folder/'tasks.json') if r['task_id'] in ids]
        languages=dict(Counter(r['language'] for r in rows))
        if set(languages)!=set(LANGUAGES):raise ValueError('Paired token filter lost a language')
        output=folder/f'training-{arm}.jsonl';content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
        recover_incomplete(output)
        if output.exists():
            if output.read_text()!=content:raise ValueError('Prepared teaching data changed')
            load_verified(output,folder/'tasks.json')
            continue
        temporary=output.with_suffix('.partial');temporary.write_text(content);temporary.replace(output)
        atomic_json(manifest_path(output),{**read(manifest_path(raw)),'dataset_sha256':sha256(output),'rows':len(rows),
            'languages':languages,'paired_filter_sha256':sha256(folder/'paired-fit.json')})
    session.save(phase='train',trial_index=0,teaching_pairs=count,teaching_examples=len(ids),
                 detail=f'Matched teaching trial: {len(ids)} examples, {count} corrected targets')


if __name__=='__main__':
    import sys
    filter_pairs(Path(sys.argv[1]))
