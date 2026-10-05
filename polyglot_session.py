"""Initial matched multilingual baseline and language-specific development reporting."""
import json
from pathlib import Path
from train_adapter import atomic_json
from training_data import sha256


def language_scores(tasks,report):
    rows={r['id']:r for r in report['tasks']}
    result={}
    for task in tasks:
        language=task.get('language','python')
        item=result.setdefault(language,{'passed':0,'evaluated':0,'total':0})
        item['total']+=1
        if task['id'] in rows:
            item['evaluated']+=1;item['passed']+=rows[task['id']].get('passed') is True
    for item in result.values():item['complete']=item['evaluated']==item['total']
    return result


def establish_baseline(session,folder):
    """A new language set needs its own complete baseline before candidate selection."""
    if session.state.get('polyglot_baseline_complete'):return True
    import learning_session as learning
    from evaluation import request
    from recursive_agent import grade,solve
    solve = learning.quality_solver(session.state)
    settings = learning.quality_settings(session.state)
    adapter=Path(session.state['best_adapter'])
    tasks=learning.development_tasks(session.path/'development-tasks.json')
    command=[str(learning.ROOT/'.cache/train-env/bin/python'),str(learning.ROOT/'research/adapter_eval_server.py'),'--adapter',str(adapter)]
    if not session.server(command,8090,folder/'initial-multilingual-baseline.log'):return False
    reports={}
    try:
        for mode in ('base','adapter'):
            file=session.path/('dev-base.json' if mode=='base' else 'dev-starting-adapter.json')
            binding={'adapter_sha256':sha256(adapter/'adapter_model.safetensors') if mode=='adapter' else None,
                     'development_tasks_sha256':sha256(session.path/'development-tasks.json'),
                     'sources':session.state['sources'],'settings':settings,'limits':learning.LIMITS}
            report=json.loads(file.read_text()) if file.exists() else {**binding,'split':'dev','tasks':[]}
            if any(report.get(k)!=v for k,v in binding.items()):raise ValueError('Multilingual baseline binding changed')
            if [r['id'] for r in report['tasks']]!=[t['id'] for t in tasks[:len(report['tasks'])]]:
                raise ValueError('Multilingual baseline progress changed')
            request('http://127.0.0.1:8090','/mode',{'mode':mode})
            if mode == 'adapter' and session.state.get('baseline_equivalent'):
                report = {**binding, 'split': 'dev', 'tasks': reports['base']['tasks'],
                          'evaluated_as': 'Verified zero-delta adapter; base outputs reused'}
                atomic_json(file, report)
            for task in tasks[len(report['tasks']):]:
                session.save(detail='Initial multilingual baseline '+mode+': '+task['id'],
                    evaluation={'mode':mode,'completed':len(report['tasks']),'total':len(tasks)})
                if session.interrupted():return False
                row=solve(task,'http://127.0.0.1:8090',**settings,**learning.LIMITS)
                if row.get('patch'):row.update(grade(task,row['patch']))
                report['tasks'].append(row);atomic_json(file,report)
            reports[mode]=report
        previous={'tasks':[{'id':r['id'],'passed':r['passed']} for r in reports['adapter']['tasks']]}
        # Preserve established Python successes rather than silently resetting the selection floor.
        required={r['id'] for r in session.state['best_dev']['tasks'] if r.get('passed')}
        actual={r['id'] for r in previous['tasks'] if r.get('passed')}
        if not required<=actual:raise ValueError('Starting adapter no longer preserves established Python passes')
        session.state['confirmation']['baseline_dev']=previous
        session.state.update(best_dev=previous,research_score=sum(r['passed'] for r in previous['tasks']),
            research_regressions=0,research_adapter=str(adapter),polyglot_baseline_complete=True,
            language_baseline=language_scores(tasks,reports['adapter']),evaluation=None)
        session.save(detail='Matched multilingual baseline complete; collecting verified teacher traces')
        return True
    finally:session.stop_child()


def prepare(controller_path,tasks_file):
    """Enable a reviewed source generation only while the native worker is idle and paused."""
    import copy
    import shutil
    import time
    import learning_session as learning
    from continuous_learning import Controller,child_busy,read
    from polyglot_runtime import LANGUAGES,image
    from training_data import registry,load_verified
    from train_adapter import verified_checkpoint
    controller=Controller(controller_path);old_path=Path(controller.state['child']);old=controller.child_state()
    if controller.desired()!='pause' or old.get('status')!='paused' or child_busy(old_path) or child_busy(controller.path):
        raise ValueError('Pause learning and end the identified idle worker before preparing a multilingual continuation')
    if old['active_seconds']>=old['limit_seconds']:raise ValueError('Do not restart a completed active budget')
    compiler=image()  # A compiler image must already exist and match its source.
    manifest=read(Path(str(tasks_file)+'.manifest.json'))
    from polyglot_data import MANIFEST
    if not manifest or manifest.get('tasks_sha256')!=sha256(tasks_file) or manifest.get('source_manifest_sha256')!=sha256(MANIFEST):
        raise ValueError('Multilingual preparation requires a verified source manifest')
    values=list(registry(tasks_file).values())
    if any(t.get('source_reference_passed') is not True or t.get('language') not in LANGUAGES for t in values):
        raise ValueError('Require independently validated multilingual train-source fixtures')
    if any(t.get('provenance',{}).get('dataset')!='deepmind/code_contests' for t in values):
        raise ValueError('Only the reviewed CodeContests training split is permitted')
    groups={}
    for task in values:
        if task['split']=='dev':groups.setdefault(task['repository'],[]).append(task)
    complete=[ts for ts in groups.values() if {t['language'] for t in ts}==set(LANGUAGES) and len(ts)==5]
    if len(complete)<9:raise ValueError('Require five development problems and four fresh confirmation problems across all languages')
    fixed=[t for ts in complete[:5] for t in ts];confirm=[t for ts in complete[5:9] for t in ts]
    from polyglot_data import rotate
    train=rotate([t for t in values if t['split']=='train'])[:256]
    if len(train)<80:raise ValueError('Require at least one full five-language training cycle')
    if {t['repository'] for t in train}&{t['repository'] for t in fixed+confirm}:
        raise ValueError('Training and evaluation problems overlap')
    child=controller.path/'children'/('polyglot-'+str(controller.state['batches']+1).zfill(6))
    if child.exists():raise ValueError('Prepared continuation already exists; inspect before repeating')
    parent_bytes=(old_path/'status.json').read_bytes()
    adapter=Path(old['best_adapter']);verified_checkpoint(adapter,read(adapter.parent/'run.json'))
    child.mkdir(parents=True)
    extra=read(old_path/'development-tasks.json',[])
    atomic_json(child/'development-tasks.json',extra+fixed)
    anchor=child/'round-001';anchor.mkdir()
    for name in ('tasks.json','teacher.json','training.jsonl','training.jsonl.manifest.json'):
        shutil.copy2(old_path/'round-001'/name,anchor/name)
    load_verified(anchor/'training.jsonl',anchor/'tasks.json')
    cumulative=list(registry(anchor/'tasks.json').values())
    batches=[train[i:i+16] for i in range(0,len(train),16)]
    for number,tasks in enumerate(batches+[[]],2):
        for task in tasks:
            value=copy.deepcopy(task);value['id']+=f'-r{number}';value['repository']+=f'-r{number}';cumulative.append(value)
        atomic_json(child/f'round-{number:03d}'/'tasks.json',cumulative)
    state=copy.deepcopy(old)
    for key in ('candidate_best','completion_reason','confirmation','continuation','recovery','consumed_confirmation'):
        state.pop(key,None)
    state.update(status='paused',phase='collect',round=2,collected=0,passed=0,training=None,evaluation=None,
        completed_rounds=[],accepted_rounds=[],rounds_without_repairs=0,pid=None,
        polyglot=True,compiler_image=compiler,polyglot_baseline_complete=False,research_adapter=str(adapter),research_regressions=0,
        sources={n:sha256(learning.ROOT/n) for n in learning.SOURCE_FILES},
        development_tasks_sha256=sha256(child/'development-tasks.json'),
        detail='Multilingual continuation prepared; accepted weights and active budget preserved')
    atomic_json(child/'confirmation-tasks.json',confirm)
    state['confirmation']={'consumed':False,'tasks':len(confirm),'reserved_before_training':True,
        'registry_sha256':sha256(child/'confirmation-tasks.json'),'baseline_adapter':str(adapter),
        'baseline_adapter_sha256':sha256(adapter/'adapter_model.safetensors'),'baseline_dev':copy.deepcopy(old['best_dev'])}
    state['continuation']={'parent_session':str(old_path),'parent_status_sha256':sha256(old_path/'status.json'),
        'parent_sources':old['sources'],'active_seconds_carried':old['active_seconds'],'limit_seconds_carried':old['limit_seconds'],
        'prepared_at':time.time(),'reason':'User requested rotating multilingual coding adaptation and standardized per-language audits'}
    state['curriculum']={'languages':list(LANGUAGES),'fresh_training_tasks':len(train),'batches':len(batches),
        'batch_size':16,'primary_examples_per_round':8,'cross_language_examples_per_round':8,
        'development_tasks':len(learning.development_tasks(child/'development-tasks.json')),
        'reserved_confirmation_tasks':20,'private_history_used':False,'tasks_sha256':sha256(tasks_file)}
    atomic_json(child/'status.json',state);atomic_json(child/'command.json',read(controller.path/'command.json'))
    if (old_path/'status.json').read_bytes()!=parent_bytes:raise ValueError('Parent session changed during preparation')
    binding=read(controller.path/'binding.json')
    from research.polyglot_benchmark import SOURCES
    names=tuple(binding)+learning.SOURCE_FILES+SOURCES+('polyglot_data.py','polyglot_session.py')
    atomic_json(controller.path/'binding.json',{n:sha256(learning.ROOT/n) for n in names})
    controller.attach_continuation(child);controller.save(last_consumed_sequence=manifest['last_consumed_sequence'])
    controller.validate()
    return child


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller',type=Path,required=True);parser.add_argument('--tasks',type=Path,required=True)
    args=parser.parse_args();print(prepare(args.controller,args.tasks))
