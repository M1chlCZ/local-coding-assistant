"""PC-local continuous supervisor around source-bound, bounded learning experiments."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from train_adapter import atomic_json, verified_checkpoint
from training_data import registry, sha256

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT/'.cache/learning/continuous'
EXPERIMENT_SECONDS = 6 * 3600


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def process_failure(log,offset,code):
    with Path(log).open('rb') as stream:
        stream.seek(max(offset,Path(log).stat().st_size-8192))
        tail=stream.read(8192).decode('utf-8',errors='replace').strip()
    return f'Local worker exited {code}; inspect {log}\n{tail}'


def child_busy(path):
    with (Path(path)/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
    return False


class Controller:
    def __init__(self,path):
        self.path=Path(path).resolve();self.state=read(self.path/'state.json')
        if not self.state:raise ValueError('Continuous learning is not configured; initialize with --adopt')

    @classmethod
    def create(cls,path,child):
        path=Path(path).resolve();child=Path(child).resolve()
        path.mkdir(parents=True,exist_ok=True)
        if (path/'state.json').exists():raise ValueError('Continuous controller already exists')
        if not (child/'status.json').exists():raise ValueError('Adoption requires a prepared learning session')
        command=read(child/'command.json',{'action':'resume'})
        atomic_json(path/'command.json',command)
        atomic_json(path/'state.json',{'schema':1,'child':str(child),'adopted':str(child),
            'status':'running','failures':0,'retry_at':0,'active_by_child':{},'batches':0,
            'detail':'Adopted prepared research; no overall active-time cutoff',
            'adopted_baseline':read(child/'status.json').get('best_dev'),
            'last_command_time':command.get('requested_at',0)})
        result=cls(path);result.snapshot(persist=True);return result

    def save(self,**changes):
        self.state.update(changes,updated_at=time.time())
        atomic_json(self.path/'state.json',self.state)

    def desired(self):
        return read(self.path/'command.json',{'action':'pause'})['action']

    def set_hours(self,hours):
        """Change an idle experiment's limit without resetting work or resuming it."""
        if not math.isfinite(hours) or not .01<=hours<=24:
            raise ValueError('Use an experiment limit between 0.01 and 24 hours')
        child=self.child_state()
        if (self.desired() not in ('pause','stop') or child_busy(self.path)
                or child_busy(self.state['child']) or child.get('status') not in ('paused','completed','stopped','failed')):
            raise ValueError('Pause learning and release the idle supervisor before changing its duration')
        seconds=hours*3600
        history=self.state.get('duration_changes',[])+[{'previous_limit_seconds':child.get('limit_seconds'),
            'limit_seconds':seconds,'active_seconds_preserved':child.get('active_seconds',0),'changed_at':time.time()}]
        self.save(experiment_limit_seconds=seconds,duration_changes=history)
        if child.get('status') not in ('completed','stopped'):
            atomic_json(Path(self.state['child'])/'status.json',{**child,'limit_seconds':seconds})
        self.snapshot(persist=True)

    def child_state(self):
        return read(Path(self.state['child'])/'status.json',{})

    def control(self,action):
        if action not in ('pause','resume','stop'):raise ValueError('Invalid continuous command')
        value={'action':action,'requested_at':time.time()}
        atomic_json(self.path/'command.json',value)
        child=self.child_state()
        if child.get('status') not in ('completed','stopped'):
            atomic_json(Path(self.state['child'])/'command.json',value)
        if action=='resume':self.state.update(status='running',failures=0,retry_at=0,detail='Resume requested')
        self.snapshot()

    def sync_manual_control(self):
        value=read(Path(self.state['child'])/'command.json',{})
        if value.get('action') in ('pause','stop') and self.desired()=='resume':
            atomic_json(self.path/'command.json',value)

    def snapshot(self,persist=False):
        child=self.child_state();key=self.state['child']
        clocks=self.state['active_by_child']
        clocks[key]=max(clocks.get(key,0),child.get('active_seconds',0))
        desired=self.desired();status=self.state['status']
        if desired=='pause':status='paused' if not self.state.get('auxiliary_active') and child.get('status') in ('paused','completed','stopped','failed') else 'pausing'
        elif desired=='stop':status='stopped' if not self.state.get('auxiliary_active') and child.get('status') in ('stopped','completed','failed') else 'stopping'
        if persist:self.save()
        value={**child,'status':status,'continuous':True,'desired':desired,
               'active_seconds':sum(clocks.values()),'limit_seconds':None,'child':key,
               'child_status':child.get('status'),'child_active_seconds':child.get('active_seconds'),
               'child_limit_seconds':child.get('limit_seconds'),'batches':self.state['batches'],
               'supervisor_detail':self.state['detail'],'retry_at':self.state['retry_at']}
        limit=child.get('limit_seconds',self.state.get('experiment_limit_seconds',EXPERIMENT_SECONDS))
        value['benchmark']={
            'schedule':f'After each experiment, at most {limit/3600:g} active training hours; unchanged weights reuse verified results',
            'next_after_active_seconds':max(0,limit-child.get('active_seconds',0)),
            'last':read(self.path/'benchmarks/latest.json'),
            'progress':read(Path(self.state['benchmark_output'])/'status.json',{'status':'starting'})
                if self.state.get('auxiliary_active') and self.state['detail'].startswith('Scheduled HumanEval') else None}
        if persist:atomic_json(self.path/'status.json',value)
        return value

    def attach(self,child):
        self.snapshot(persist=True);self.save(child=str(Path(child).resolve()),status='running',failures=0,retry_at=0)

    def attach_continuation(self,child):
        """Move the carried clock to a new source generation, without counting it twice."""
        value=self.snapshot(persist=True);old=self.state['child'];child=str(Path(child).resolve())
        new=read(Path(child)/'status.json')
        if new['active_seconds']!=value['child_active_seconds'] or new.get('limit_seconds')!=self.child_state().get('limit_seconds'):
            raise ValueError('Continuation must preserve active time and budget')
        history=self.state.get('continuations',[])+[{'parent':old,'child':child,
            'active_seconds_carried':new['active_seconds'],'prepared_at':time.time()}]
        clocks=dict(self.state['active_by_child']);clocks[child]=clocks.pop(old)
        self.save(child=child,active_by_child=clocks,continuations=history,status='running',failures=0,retry_at=0,polyglot=True)

    def failure(self,detail,now=None):
        now=time.time() if now is None else now
        failures=self.state['failures']+1
        fatal=any(word in detail.lower() for word in ('integrity','digest','source changed','registry changed',
                  'checks changed','manifest changed','different data','different training','exhausted'))
        temporary=bool(re.search(r'curl: \((?:5|6|7|18|28|35|52|55|56)\)|'
            r'temporary failure in name resolution|connection (?:reset|refused|aborted)|network is unreachable|'
            r'(?:http error |requested url returned error: )(?:408|429|502|503|504)',detail,re.I))
        temporary=temporary or ('timed out' in detail.lower() and any(name in detail for name in
            ('urllib','http.client','requests.exceptions','httpx.')))
        self.save(failures=failures,status='blocked' if fatal or (failures>=6 and not temporary) else 'waiting',
                  retry_at=now+min(900,30*2**min(failures-1,5)),detail=detail)

    def should_launch(self,now=None):
        now=time.time() if now is None else now
        return (self.desired()=='resume' and self.state['status']!='blocked'
                and now>=self.state['retry_at'] and self.child_state().get('status') not in ('completed','stopped'))

    def validate(self):
        import learning_session as learning
        child=self.child_state()
        sources={name:sha256(ROOT/name) for name in learning.SOURCE_FILES}
        binding=read(self.path/'binding.json')
        archived=self.state.get('completed_source_generations',{}).get(self.state['child'])
        if child.get('status')=='completed' and archived:
            directory=Path(archived['directory']).resolve()
            if (not directory.is_relative_to(self.path/'source-updates') or archived['sources']!=child.get('sources')
                    or not binding or any(binding.get(name)!=digest for name,digest in sources.items())
                    or any(sha256(directory/name)!=digest for name,digest in archived['sources'].items())):
                raise ValueError('Completed session source changed; restore the archived trainer')
        elif child.get('sources')!=sources:
            raise ValueError('Session source changed; restore the bound trainer before resuming')
        if child.get('polyglot'):
            from polyglot_runtime import image
            if image()!=child.get('compiler_image'):raise ValueError('Compiler image changed; review before continuing')
        if binding and any(sha256(ROOT/name)!=digest for name,digest in binding.items()):
            raise ValueError('Continuous source manifest changed; review before continuing')

    def benchmark_target(self):
        child=self.child_state()
        confirmation=read(self.path/'confirmations'/Path(self.state['child']).name/'summary.json',{})
        if child.get('status')!='completed' or not confirmation.get('completed'):
            raise ValueError('Scheduled audit requires a completed experiment and fresh confirmation')
        adapter=Path(child['confirmation']['baseline_adapter'] if confirmation.get('regressions') else child['best_adapter'])
        digest=sha256(adapter/'adapter_model.safetensors')
        return adapter,self.path/'benchmarks'/digest

    def audit_command(self):
        adapter,output=self.benchmark_target()
        polyglot=self.child_state().get('polyglot')
        if polyglot:output=self.path/'polyglot-benchmarks'/output.name
        runner='research/polyglot_benchmark.py' if polyglot else 'research/student_benchmark.py'
        executable=str(ROOT/'.cache/data-env/bin/python') if polyglot else sys.executable
        prior=Path(self.state.get('benchmark_output') or output)
        archive=self.archived_audit(prior)
        if archive and read(prior/'binding.json')['adapter_sha256']==sha256(adapter/'adapter_model.safetensors'):
            command=[executable,str(archive/runner),'--adapter',str(adapter),'--output',str(prior)]
            size=read(prior/'binding.json').get('generation_batch_size',1)
            if polyglot and size>1:command+=['--batch-size',str(size)]
            return command,prior
        command=[executable,str(ROOT/runner),'--adapter',str(adapter)]
        size=self.state.get('generation_batch_size',1) if polyglot else 1
        if type(size) is not int or size not in (1,2,4,8,16):raise ValueError('Use a checked generation batch size')
        if size>1:
            from research.polyglot_benchmark import SOURCES
            digest=hashlib.sha256(json.dumps({n:sha256(ROOT/n) for n in SOURCES},sort_keys=True).encode()).hexdigest()[:12]
            output=output.with_name(output.name+f'-batch{size}-'+digest)
            command+=['--output',str(output),'--batch-size',str(size)]
        else:command+=['--output',str(output)]
        return command,output

    def archived_audit(self,output):
        archive=self.state.get('completed_source_generations',{}).get(self.state['child'],{})
        binding=read(Path(output)/'binding.json',{})
        if not archive.get('benchmark_sources') or binding.get('sources')!=archive['benchmark_sources']:return None
        directory=Path(archive['directory']).resolve()
        if (not directory.is_relative_to(self.path/'source-updates') or not Path(output).resolve().is_relative_to(self.path)
                or any(sha256(directory/name)!=digest for name,digest in archive['benchmark_sources'].items())):
            raise ValueError('Archived audit source changed')
        return directory

    def record_polyglot_benchmark(self,output,reused):
        from research.polyglot_benchmark import SOURCES,load_tasks,summarize
        _,tasks=load_tasks();binding=read(output/'binding.json')
        if ((binding['sources']!={n:sha256(ROOT/n) for n in SOURCES} and not self.archived_audit(output))
                or binding.get('smoke_limit')):
            raise ValueError('Multilingual audit source changed or report is partial')
        reports={m:{l:read(output/(m+'-'+l+'.json')) for l in binding['languages']} for m in ('base','adapter')}
        summary=summarize(binding,tasks,reports)
        if summary!=read(output/'summary.json'):raise ValueError('Multilingual benchmark summary changed')
        adapter,_=self.benchmark_target()
        if binding['adapter_sha256']!=sha256(adapter/'adapter_model.safetensors'):
            raise ValueError('Multilingual audit belongs to different weights')
        languages=summary['languages']
        atomic_json(self.path/'benchmarks/latest.json',{
            'benchmark':'HumanEval / MultiPL-E','languages':languages,'macro_pass_at_1':summary['macro_pass_at_1'],
            'adapter_sha256':binding['adapter_sha256'],'experiment':Path(self.state['child']).name,
            'reported_at':time.time(),'active_seconds':self.snapshot()['active_seconds'],'reused':reused,
            'base_passed':sum(v['models']['base']['passed'] for v in languages.values()),
            'adapter_passed':sum(v['models']['adapter']['passed'] for v in languages.values()),
            'total':sum(v['models']['base']['total'] for v in languages.values()),
            'gained':sum(len(v['gained']) for v in languages.values()),'lost':sum(len(v['lost']) for v in languages.values()),
            'report':str(output/'summary.json'),'purpose':'Reporting only; no training or checkpoint selection'})

    def record_benchmark(self,output,reused):
        from research.student_benchmark import SOURCES,load_report,load_tasks,summarize
        binding=read(output/'binding.json',{})
        if (binding.get('adapter_sha256')!=output.name
                or (binding.get('sources')!={name:sha256(ROOT/name) for name in SOURCES} and not self.archived_audit(output))):
            raise ValueError('Scheduled benchmark report failed source or adapter integrity checks')
        _,tasks=load_tasks()
        reports={name:load_report(output/f'{name}.json',binding,tasks) for name in ('base','adapter')}
        summary=summarize(binding,tasks,reports)
        if read(output/'summary.json')!=summary:
            raise ValueError('Scheduled benchmark summary does not match complete task reports')
        comparison=summary['adapter_vs_base'];models=summary['models']
        atomic_json(self.path/'benchmarks/latest.json',{
            'adapter_sha256':binding['adapter_sha256'],'experiment':Path(self.state['child']).name,
            'reported_at':time.time(),'active_seconds':self.snapshot()['active_seconds'],'reused':reused,
            'base_passed':models['base']['passed'],'adapter_passed':models['adapter']['passed'],
            'total':len(tasks),'gained':len(comparison['gained']),'lost':len(comparison['lost']),
            'base_median_seconds':models['base']['median_model_answer_seconds'],
            'adapter_median_seconds':models['adapter']['median_model_answer_seconds'],
            'report':str(output/'summary.json'),'purpose':'Reporting only; no training or checkpoint selection'})

    def stash_adapter(self,value):
        path=Path(value);binding=read(path.parent/'run.json')
        verified_checkpoint(path,binding)
        digest=sha256(path/'adapter_model.safetensors')
        target=self.path/'models'/digest/'checkpoint'
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name('checkpoint.partial')
            shutil.rmtree(temporary,ignore_errors=True)
            shutil.copytree(path,temporary)
            atomic_json(target.parent/'run.json',binding)
            temporary.rename(target)
        verified_checkpoint(target,binding)
        return str(target)

    def pending_tasks(self):
        for folder in sorted((self.path/('polyglot-pool' if self.state.get('polyglot') else 'pool')).glob('batch-*')):
            if int(folder.name[6:])>self.state.get('last_consumed_sequence',0) and (folder/'manifest.json').exists():
                return folder/'tasks.json'
        return None

    def prepare_child(self,tasks_file):
        import learning_session as learning
        parent=Path(self.state['child']);old=self.child_state();self.validate()
        if tasks_file.parent.name.startswith('batch-'):
            if self.state.get('polyglot'):
                from polyglot_data import MANIFEST
            else:
                from continuous_data import MANIFEST
            manifest=read(tasks_file.parent/'manifest.json')
            if manifest['source_manifest_sha256']!=sha256(MANIFEST) or manifest['tasks_sha256']!=sha256(tasks_file):
                raise ValueError('Queued curriculum digest failed integrity verification')
        all_tasks=list(registry(tasks_file).values()) if read(tasks_file) else []
        train=[t for t in all_tasks if t['split']=='train'][:256]
        direct = old.get('recipe') == 'balanced-code-v1'
        if direct:
            from code_recipe import balanced_order
            train = balanced_order([t for t in all_tasks if t['split']=='train'], limit=250)
        excluded={t['repository'] for t in read(parent/'development-tasks.json',[])}
        confirm=[t for t in all_tasks if t['split']=='dev' and t['repository'] not in excluded][:20]
        if len(train)<16 or len(confirm)<20:
            self.save(last_consumed_sequence=int(tasks_file.parent.name[6:]) if tasks_file.parent.name.startswith('batch-') else self.state.get('last_consumed_sequence',0),detail='Validated window too small; advancing to another public window')
            return None
        number=self.state['batches']+1;child=self.path/'children'/f'batch-{number:06d}'
        # Pending preparation is recoverable; it has never owned a worker or GPU.
        if child.exists():shutil.rmtree(child)
        child.mkdir(parents=True)
        if (parent/'development-tasks.json').exists():
            shutil.copy2(parent/'development-tasks.json',child/'development-tasks.json')
        shutil.copy2(parent/'dev-base.json',child/'dev-base.json')
        anchor=child/'round-001';anchor.mkdir()
        if direct and old.get('recipe_trials'):
            from code_recipe import replay_anchor
            replay_anchor([parent],anchor)
        else:
            for name in ('tasks.json','teacher.json','training.jsonl','training.jsonl.manifest.json'):
                if not direct or (parent/'round-001'/name).exists():
                    shutil.copy2(parent/'round-001'/name,anchor/name)
        cumulative=list(registry(anchor/'tasks.json').values())
        batch_size = (100 if old.get('recipe_trials') else 20) if direct else 16
        batches=[train[i:i+batch_size] for i in range(0,len(train),batch_size)]
        for index,tasks in enumerate(batches+[[]],2):
            for original in tasks:
                task=copy.deepcopy(original);task['id']+=f'-r{index}';task['repository']+=f'-r{index}';cumulative.append(task)
            atomic_json(child/f'round-{index:03d}'/'tasks.json',cumulative)
        state=copy.deepcopy(old)
        comparison=read(self.path/'confirmations'/parent.name/'summary.json',{})
        if comparison.get('regressions'):
            state['best_adapter']=old['confirmation']['baseline_adapter']
            state['research_adapter']=state['best_adapter']
            state['best_dev']=old['confirmation'].get('baseline_dev') or self.state['adopted_baseline']
            state['research_score']=sum(r['passed'] for r in state['best_dev']['tasks'])
            state['research_regressions']=0
        for key in ('completion_reason','candidate_best','continuation','confirmation','consumed_confirmation','recovery',
                    'trial_index','trial_origin_adapter','trial_origin_dev','teaching_trial','teaching_pairs','teaching_examples'):
            state.pop(key,None)
        for field in ('best_adapter','research_adapter'):
            if state.get(field):state[field]=self.stash_adapter(state[field])
        state.update(status='paused',phase='collect',round=2,active_seconds=0,
            limit_seconds=self.state.get('experiment_limit_seconds',EXPERIMENT_SECONDS),
            sources={name:sha256(ROOT/name) for name in learning.SOURCE_FILES},training_retry=None,recovered_examples=0,
            completed_rounds=[],accepted_rounds=[],collected=0,passed=0,training=None,evaluation=None,
            rounds_without_repairs=0,pid=None,detail='Fresh continuous curriculum; accepted model preserved')
        if direct:
            size=self.state.get('generation_batch_size',old.get('generation_batch_size',1))
            if size!=old.get('generation_batch_size',1):
                # Batched floating-point paths need a fresh matched baseline, never a mixed comparison.
                (child/'dev-base.json').rename(child/'previous-serial-baseline.json')
                state.update(generation_batch_size=size,polyglot_baseline_complete=False,best_dev={'tasks':[]},
                    research_score=0,research_regressions=0,language_baseline=None,
                    detail='New inference batch size; establishing its matched baseline')
        atomic_json(child/'confirmation-tasks.json',confirm)
        state['confirmation']={'consumed':False,'tasks':len(confirm),'reserved_before_training':True,
            'registry_sha256':sha256(child/'confirmation-tasks.json'),
            'baseline_dev':copy.deepcopy(state.get('best_dev')),
            'baseline_adapter':state['best_adapter'],
            'baseline_adapter_sha256':sha256(Path(state['best_adapter'])/'adapter_model.safetensors')}
        atomic_json(child/'status.json',state)
        atomic_json(child/'command.json',{'action':'resume','requested_at':time.time()})
        atomic_json(child/'curriculum.json',{'source_tasks_sha256':sha256(tasks_file),'training_tasks':len(train),
            'reserved_confirmation_tasks':len(confirm),'training_used_private_history':False})
        self.attach(child);self.save(batches=number,last_consumed_sequence=int(tasks_file.parent.name[6:]) if tasks_file.parent.name.startswith('batch-') else self.state.get('last_consumed_sequence',0),detail='Fresh checked public curriculum ready')
        self.prune()
        return child

    def prune(self):
        # Delete only completed, controller-owned children. Adopted/pre-existing sessions stay intact.
        children=sorted((self.path/'children').glob('batch-*'))
        for path in children[:-2]:
            if str(path)==self.state['child'] or read(path/'status.json',{}).get('status')!='completed':continue
            if child_busy(path):continue
            atomic_json(self.path/'history'/f'{path.name}.json',read(path/'status.json'))
            shutil.rmtree(path)
        protected=set()
        for path in children[-2:]+[Path(self.state['child'])]:
            s=read(path/'status.json',{})
            protected.update(s.get(field) for field in ('best_adapter','research_adapter'))
            protected.add(s.get('confirmation',{}).get('baseline_adapter'))
        for folder in (self.path/'models').glob('*'):
            if str(folder/'checkpoint') not in protected:shutil.rmtree(folder)
        # Source batches are already copied into child registries; retain two newest windows.
        for folder in sorted((self.path/('polyglot-pool' if self.state.get('polyglot') else 'pool')).glob('batch-*'))[:-2]:shutil.rmtree(folder)

    def run_process(self,command,log,forward=True):
        # Existing Session ownership records protect model children across a supervisor crash.
        self.save(auxiliary_active=not forward)
        try:
            with Path(log).open('ab') as stream:
                offset=stream.tell()
                process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                                         env={**os.environ,'LCA_CONTINUOUS_OWNER':str(self.path)})
        except OSError:
            self.save(auxiliary_active=False)
            raise
        start=Path(f'/proc/{process.pid}/stat').read_text().split(') ',1)[1].split()[19]
        atomic_json(self.path/'process.json',{'pid':process.pid,'start':start,'forward':forward})
        try:
            while process.poll() is None:
                self.sync_manual_control()
                if self.desired()!='resume':
                    if forward:
                        atomic_json(Path(self.state['child'])/'command.json',read(self.path/'command.json'))
                    else:
                        os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=30)
                        return False
                self.snapshot(persist=True);time.sleep(2)
            if process.returncode:raise RuntimeError(process_failure(log,offset,process.returncode))
            return True
        finally:
            self.save(auxiliary_active=False)
            (self.path/'process.json').unlink(missing_ok=True)
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()

    def recover_process(self):
        ownership=read(self.path/'process.json')
        if not ownership:return
        # A surviving source-bound trainer owns its own recovery and GPU lock.
        if not ownership['forward']:
            proc=Path('/proc')/str(ownership['pid'])
            try:
                start=proc.joinpath('stat').read_text().split(') ',1)[1].split()[19]
                env=proc.joinpath('environ').read_bytes().split(b'\0')
                if start==ownership['start'] and ('LCA_CONTINUOUS_OWNER='+str(self.path)).encode() in env:
                    os.killpg(ownership['pid'],signal.SIGTERM)
            except (FileNotFoundError,ProcessLookupError):pass
        (self.path/'process.json').unlink(missing_ok=True)
        self.save(auxiliary_active=False)

    def run(self):
        self.path.mkdir(exist_ok=True)
        with (self.path/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.run_locked()

    def run_locked(self):
        try:self.validate()
        except (OSError,ValueError) as error:self.failure(str(error))
        self.recover_process()
        while True:
            self.sync_manual_control()
            command=read(self.path/'command.json')
            if command.get('requested_at',0)>self.state.get('last_command_time',0):
                changes={'last_command_time':command.get('requested_at',0)}
                if command['action']=='resume':changes.update(status='running',failures=0,retry_at=0)
                self.save(**changes)
            value=self.snapshot(persist=True)
            if self.desired()=='stop':
                if not child_busy(self.state['child']):return
            if self.desired()!='resume' or self.state['status']=='blocked' or time.time()<self.state['retry_at']:
                time.sleep(2);continue
            child=Path(self.state['child']);status=self.child_state()
            if child_busy(child):time.sleep(2);continue  # Adopt the Windows-owned child; never duplicate it.
            try:
                self.validate()
                if status['status']=='stopped':
                    self.control('stop');continue
                if status['status']=='completed':
                    self.save(status='running',detail='Completing reserved matched confirmation checks')
                    command=[sys.executable,str(ROOT/'continuous_confirmation.py'),'--controller',str(self.path)]
                    if not self.run_process(command,self.path/'confirmation.log',forward=False):continue
                    command,output=self.audit_command()
                    reused=(output/'summary.json').exists()
                    self.save(benchmark_output=str(output),detail='Scheduled HumanEval / MultiPL-E comparison; complete reports required')
                    if not self.run_process(command,self.path/'benchmark.log',forward=False):continue
                    if status.get('polyglot'):self.record_polyglot_benchmark(output,reused)
                    else:self.record_benchmark(output,reused)
                    tasks=self.pending_tasks()
                    if tasks is None:
                        self.save(detail='Preparing fresh pinned public fixtures; no GPU needed')
                        command=[str(ROOT/'.cache/data-env/bin/python'),str(ROOT/('polyglot_data.py' if self.state.get('polyglot') else 'continuous_data.py')),'--pool',str(self.path/('polyglot-pool' if self.state.get('polyglot') else 'pool'))]
                        if not self.run_process(command,self.path/'curriculum.log',forward=False):continue
                        tasks=self.pending_tasks()
                    if tasks is None:raise ValueError('Prepared curriculum is missing')
                    self.prepare_child(tasks)
                elif self.should_launch():
                    self.save(status='running',detail='Running source-bound CUDA research')
                    if status['status']=='failed':
                        # Source/checkpoint failures remain blocked; transient child failures can recover.
                        detail=status.get('detail','')
                        if any(word in detail.lower() for word in ('integrity','digest','changed','different')):
                            raise ValueError(detail)
                    self.run_process([sys.executable,str(ROOT/'learning_session.py'),'run','--session',str(child),
                                      '--research','--fast-reject'],self.path/'worker.log')
                    self.save(failures=0,retry_at=0)
            except (OSError,ValueError,RuntimeError) as error:
                detail=self.child_state().get('detail','') if self.child_state().get('status')=='failed' else ''
                self.failure(detail or str(error))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('init','run','status','pause','resume','stop','set-hours'))
    parser.add_argument('--session',type=Path,default=DEFAULT)
    parser.add_argument('--adopt',type=Path)
    parser.add_argument('--start-row',type=int,default=0)
    parser.add_argument('--hours',type=float,default=6)
    args=parser.parse_args()
    try:
        if args.action=='init':
            if not args.adopt:raise ValueError('Pass --adopt for a prepared source-bound session')
            from training_data import load_verified
            seed=read(args.adopt/'status.json',{})
            if not seed.get('best_adapter'):raise ValueError('Adoption requires an accepted adapter')
            load_verified(args.adopt/'round-001/training.jsonl',args.adopt/'round-001/tasks.json')
            adapter=Path(seed['best_adapter']);verified_checkpoint(adapter,read(adapter.parent/'run.json'))
            if not (args.adopt/'dev-base.json').exists():raise ValueError('Adoption requires a completed base evaluation')
            controller=Controller.create(args.session,args.adopt)
            from continuous_data import initialize
            initialize(controller.path/'pool',row=args.start_row)
            names=('continuous_learning.py','continuous_data.py','continuous_confirmation.py','learning_bootstrap.py','research/opencode-source.json')
            from research.student_benchmark import SOURCES
            names+=SOURCES+('research/humaneval_manifest.json','research/HumanEval.jsonl.gz')
            import learning_session as learning
            names+=learning.SOURCE_FILES
            atomic_json(controller.path/'binding.json',{n:sha256(ROOT/n) for n in names})
            print(json.dumps(controller.snapshot()));return
        controller=Controller(args.session)
        if args.action=='run':
            from learning_bootstrap import check_ready, RETRY_EXIT
            ready,reason=check_ready(controller.path,continuous=True)
            if not ready:parser.exit(RETRY_EXIT,reason+'; retry in a fresh WSL process\n')
            controller.run()
        elif args.action=='status':print(json.dumps(controller.snapshot()))
        elif args.action=='set-hours':controller.set_hours(args.hours);print(json.dumps(controller.snapshot()))
        else:controller.control(args.action);print(json.dumps(controller.snapshot()))
    except (ValueError,OSError,RuntimeError) as error:parser.exit(1,str(error)+'\n')


if __name__=='__main__':main()
