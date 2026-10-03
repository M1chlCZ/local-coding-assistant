"""Finite, checkpointed local learning with Windows controls and one GPU owner."""
import argparse
import copy
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path, PureWindowsPath

from train_adapter import atomic_json, checkpoint_binding, latest_checkpoint, verified_checkpoint, parser as training_parser
from training_data import export, load_verified, manifest_path, sha256, registry

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen3-4B'
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
LIMITS = {'calls': 8, 'output_tokens': 4096, 'seconds': 120}
SETTINGS = {'mode': 'rlm', 'depth': 2, 'instruction': ''}
DEFAULT = ROOT/'.cache/learning/tuned'
SOURCE_FILES = ('learning_session.py', 'train_adapter.py', 'training_data.py',
                'recursive_agent.py', 'rlm_worker.py', 'research/adapter_eval_server.py',
                'launcher.py', 'wsl_server.py', 'research/repair_tasks.json')


def development_tasks(extra=None):
    originals = json.loads((ROOT/'research/rlm_tasks.json').read_text())
    added = json.loads((ROOT/'research/repair_tasks.json').read_text())
    tasks = [t for t in originals+added if t['split']=='dev']
    if extra is not None:
        added = list(registry(extra).values())
        if len(added)>50 or any(t['split']!='dev' for t in added):
            raise ValueError('Extra quality checks require at most 50 development tasks')
        tasks += added
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('Development task IDs must be unique')
    return tasks


def round_tasks(index):
    """Thirty distinct training repairs with visible tests; no evaluation repositories."""
    originals = json.loads((ROOT/'research/rlm_tasks.json').read_text())
    added = json.loads((ROOT/'research/repair_tasks.json').read_text())
    tasks = []
    for source in originals+added:
        if source['split'] != 'train':
            continue
        task = copy.deepcopy(source)
        task['id'] += f'-r{index}'
        task['repository'] += f'-r{index}'
        if 'test_visible.py' not in task['files']:
            # Existing authored training checks are visible; dev and holdout checks remain excluded.
            task['files']['test_visible.py'] = task['checks']+"\nprint('VISIBLE_CHECKS_PASSED')\n"
            task['prompt'] += ' Run test_visible.py before and after the repair. Keep the test file unchanged.'
        tasks.append(task)
    return tasks


def should_stop_for_quality(results, research=False):
    return not research and len(results)>=3 and not any(r['accepted'] for r in results[-3:])


def accepted(candidate, baseline, previous=None, expected=5):
    """All development tasks must complete; require more passes and no per-task regression."""
    def passes(report):
        return {r['id']: r.get('passed') is True for r in report['tasks']}
    new, base = passes(candidate), passes(baseline)
    prior = passes(previous) if previous else base
    return (len(new) == expected and new.keys() == base.keys() == prior.keys()
            and sum(new.values()) > max(sum(base.values()), sum(prior.values()))
            and all(new[k] for k in new if base[k] or prior[k]))


def cannot_improve(rows, baseline, previous=None, research_best=None):
    """Reject a prefix only when its best possible completion cannot be retained."""
    base = {r['id']:r.get('passed') is True for r in baseline['tasks']}
    prior = {r['id']:r.get('passed') is True for r in (previous or baseline)['tasks']}
    seen = {r['id']:r.get('passed') is True for r in rows}
    if len(seen)!=len(rows) or prior.keys()!=base.keys() or not seen.keys()<=base.keys():
        raise ValueError('Early rejection requires matching unique development task IDs')
    ceiling = sum(seen.values()) + len(base)-len(seen)
    lost = sum(prior[k] and not seen[k] for k in seen)
    promotion = (ceiling>max(sum(base.values()),sum(prior.values()))
                 and all(seen[k] for k in seen if base[k] or prior[k]))
    research = research_best is not None and lost<=1 and (ceiling,-lost)>(research_best[0],-research_best[1])
    return not (promotion or research)


class Session:
    def __init__(self, path, hours, model, research=False, fast_reject=False):
        self.path = path.resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        # ponytail: one worker per session, one GPU; add a job scheduler only for multiple GPUs.
        self.lock = (self.path/'worker.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (ROOT/'.cache/learning').mkdir(parents=True, exist_ok=True)
        self.gpu_lock = (ROOT/'.cache/learning/gpu.lock').open('a')
        fcntl.flock(self.gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {name: sha256(ROOT/name) for name in SOURCE_FILES}
        self.state = json.loads((self.path/'status.json').read_text()) if (self.path/'status.json').exists() else {
            'schema_version': 1, 'status': 'running', 'phase': 'collect', 'round': 1,
            'active_seconds': 0, 'limit_seconds': hours*3600, 'teacher_model': str(model.resolve()), 'research': research,
            'fast_reject': fast_reject,
            'best_adapter': None, 'accepted_rounds': [], 'completed_rounds': [],
            'tasks_sha256': sha256(ROOT/'research/rlm_tasks.json'), 'sources': sources,
            'development_tasks_sha256': sha256(self.path/'development-tasks.json')
                if (self.path/'development-tasks.json').exists() else None}
        if self.state['tasks_sha256'] != sha256(ROOT/'research/rlm_tasks.json'):
            raise ValueError('Original task registry changed; use a new session')
        if self.state.get('sources') != sources:
            raise ValueError('Session source changed; restore the original source or use a new session')
        extra = self.path/'development-tasks.json'
        if self.state.get('development_tasks_sha256') != (sha256(extra) if extra.exists() else None):
            raise ValueError('Development checks changed; use a new session')
        if extra.exists():
            development_tasks(extra)
        self.child = None
        self.clock = time.monotonic()
        self.recover_child()

    def recover_child(self):
        ownership = self.path/'child.json'
        if not ownership.exists():
            return
        value = json.loads(ownership.read_text())
        proc = Path('/proc')/str(value['pid'])
        try:
            start = proc.joinpath('stat').read_text().split(') ', 1)[1].split()[19]
            env = proc.joinpath('environ').read_bytes().split(b'\0')
            if start == value['start'] and ('LCA_LEARNING_SESSION='+str(self.path)).encode() in env:
                os.killpg(value['pid'], signal.SIGTERM)
                time.sleep(2)
                if proc.exists():
                    os.killpg(value['pid'], signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError):
            pass
        ownership.unlink(missing_ok=True)

    def save(self, **changes):
        now = time.monotonic()
        if self.state['status'] != 'paused':
            self.state['active_seconds'] += now-self.clock
        self.clock = now
        self.state.update(changes, pid=os.getpid(), updated_at=time.time())
        atomic_json(self.path/'status.json', self.state)

    def command(self):
        file = self.path/'command.json'
        return json.loads(file.read_text())['action'] if file.exists() else 'resume'

    def interrupted(self):
        return self.command() in ('pause', 'stop') or self.state['active_seconds'] >= self.state['limit_seconds']

    def stop_child(self):
        if self.child:
            if self.child.poll() is None:
                os.killpg(self.child.pid, signal.SIGTERM)
                try: self.child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(self.child.pid, signal.SIGKILL)
                    self.child.wait()
            self.child = None
        (self.path/'child.json').unlink(missing_ok=True)

    def spawn(self, command, log):
        if self.child:
            raise ValueError('Another session child still owns the GPU')
        env = {**os.environ, 'PYTHONPATH': str(ROOT), 'HF_HUB_OFFLINE': '1',
               'LCA_LEARNING_SESSION': str(self.path)}
        with Path(log).open('ab') as output:
            self.child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                          stderr=subprocess.STDOUT, start_new_session=True)
        start = Path(f'/proc/{self.child.pid}/stat').read_text().split(') ',1)[1].split()[19]
        atomic_json(self.path/'child.json', {'pid': self.child.pid, 'start': start})

    def server(self, command, port, log):
        from launcher import health
        import socket
        # Refuse occupied ports, including an unrelated server that is still loading.
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise ValueError(f'Port {port} is occupied. Stop that server before learning.')
        self.spawn(command, log)
        deadline = time.monotonic()+180
        while not health(port):
            self.save(detail='Loading model')
            if self.child.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Model server failed to start; see '+str(log))
            if self.interrupted():
                return False
            time.sleep(1)
        return True

    def training_command(self, round_path):
        warm = self.state.get('research_adapter') if self.state.get('research') else None
        warm = warm or self.state['best_adapter']
        steps = (5, 10, 20)[(self.state['round']-2)%3] if self.state.get('research') and warm else (20 if warm else 80)
        command = [str(ROOT/'.cache/train-env/bin/python'), str(ROOT/'train_adapter.py'),
            '--dataset', str(round_path/'training.jsonl'), '--tasks', str(round_path/'tasks.json'),
            '--model', MODEL, '--revision', REVISION, '--output', str(round_path/'adapter'),
            '--max-steps', str(steps), '--max-epochs', '1', '--learning-rate',
            '0.000005' if warm else '0.00005',
            '--max-length', '4096', '--save-steps', '5',
            '--pause-file', str(self.path/'pause-training')]
        if warm:
            command += ['--warm-start', warm]
        return command

    def teacher_baseline(self, folder):
        """Measure the larger compatible model once before the first student update."""
        from recursive_agent import grade, solve
        path = folder/'teacher-dev.json'
        report = json.loads(path.read_text()) if path.exists() else {'tasks': [], 'split': 'dev',
            'sources': self.state['sources'], 'settings': SETTINGS, 'limits': LIMITS}
        tasks = development_tasks(self.path/'development-tasks.json') if self.state.get('development_tasks_sha256') else development_tasks()
        for task in tasks[len(report['tasks']):]:
            self.save(detail='Larger model development check: '+task['id'])
            if self.interrupted():
                return False
            row = solve(task, 'http://127.0.0.1:8080', **SETTINGS, **LIMITS)
            if row.get('patch'):
                row.update(grade(task, row['patch']))
            report['tasks'].append(row)
            atomic_json(path, report)
        self.save(teacher_dev_passed=sum(r['passed'] for r in report['tasks']), teacher_dev_total=len(report['tasks']))
        return True

    def collect(self, folder):
        from recursive_agent import grade, solve
        tasks_path = folder/'tasks.json'
        if not tasks_path.exists():
            old = [] if self.state['round'] == 1 else json.loads(
                (self.path/f"round-{self.state['round']-1:03d}/tasks.json").read_text())
            atomic_json(tasks_path, old + round_tasks(self.state['round']))
        tasks = json.loads(tasks_path.read_text())
        current = [t for t in tasks if t['id'].endswith(f"-r{self.state['round']}")]
        report_path = folder/'teacher.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {
            'schema_version': 1, 'split': 'train', 'tasks_sha256': sha256(tasks_path),
            'settings': SETTINGS, 'limits': LIMITS, 'tasks': []}
        if len(report['tasks']) == len(current):
            self.save(phase='export', detail='Teacher collection complete')
            return
        command = [str(ROOT/'.cache/rlm-env/bin/python'), str(ROOT/'wsl_server.py'),
                   '--model', self.state['teacher_model']]
        if not self.server(command, 8080, folder/'teacher.log'):
            return
        if self.state['round']==1 and not self.teacher_baseline(folder):
            return
        for task in current[len(report['tasks']):]:
            self.save(detail='Teacher task '+task['id'])
            if self.interrupted():
                return
            # Reference repairs validate fixtures, never supply a training answer.
            reference = grade(task, task['reference_patch'])
            if not reference['passed']:
                feedback = reference.get('feedback', '')
                if (task.get('provenance', {}).get('dataset')!='nvidia/OpenCodeInstruct'
                        or not feedback.startswith('Traceback (most recent call last):')):
                    raise ValueError('Reference failed: '+task['id'])
                row = {key:task[key] for key in ('id','repository','split')}
                row.update(mode='rlm', depth=2, passed=False, trace=[], quarantined=True,
                           feedback=feedback, error='Public reference tests failed; excluded from training')
            else:
                if grade(task, {p: task['files'][p] for p in task['editable']})['passed']:
                    raise ValueError('Broken task already passes: '+task['id'])
                row = solve(task, 'http://127.0.0.1:8080', **SETTINGS, **LIMITS)
                if row.get('patch'):
                    row.update(grade(task, row['patch']))
            report['tasks'].append(row)
            atomic_json(report_path, report)
            self.save(collected=len(report['tasks']), passed=sum(r['passed'] for r in report['tasks']))
            if self.interrupted():
                return
        self.save(phase='export', detail='Teacher collection complete')

    def dataset(self, folder):
        report = json.loads((folder/'teacher.json').read_text())
        if not any(row.get('passed') for row in report['tasks']):
            failures = self.state.get('rounds_without_repairs', 0)+1
            self.save(rounds_without_repairs=failures, round=self.state['round']+1, phase='collect',
                      status='completed' if failures>=3 else 'running',
                      detail='No verified new repairs; stopped after three empty rounds' if failures>=3
                      else 'No verified new repairs; collecting another round')
            return
        self.state['rounds_without_repairs'] = 0
        dataset = folder/'training.jsonl'
        if dataset.exists() and not manifest_path(dataset).exists():
            dataset.rename(folder/f'training-incomplete-{time.time_ns()}.jsonl')
        if not dataset.exists():
            # Bind earlier reports to the cumulative registry without modifying their originals.
            reports = []
            digest = sha256(folder/'tasks.json')
            # Research keeps its original verified anchor rather than diluting it with recent rounds.
            rounds = sorted({1, self.state['round']}) if self.state.get('research') else range(max(1, self.state['round']-3), self.state['round']+1)
            for number in rounds:
                original = self.path/f'round-{number:03d}/teacher.json'
                value = json.loads(original.read_text())
                value['tasks_sha256'] = digest
                value['original_source_sha256'] = sha256(original)
                path = folder/f'source-{number:03d}.json'
                atomic_json(path, value)
                reports.append(path)
            # Preserve the successful teacher's actions and their exact REPL feedback.
            export(reports, folder/'tasks.json', dataset, repairs=False)
        rows = load_verified(dataset, folder/'tasks.json')
        previous = self.path/f"round-{self.state['round']-1:03d}"
        if self.state['round'] > 1 and (previous/'training.jsonl').exists():
            old = load_verified(previous/'training.jsonl', previous/'tasks.json')
            current_messages = {json.dumps(row['messages'], sort_keys=True) for row in rows}
            old_messages = {json.dumps(row['messages'], sort_keys=True) for row in old}
            if current_messages <= old_messages:
                self.save(status='completed', detail='No new verified examples; stopped before duplicate training')
                return
        self.save(phase='train', detail='Dataset fixed for checkpoint resume')

    def train(self, folder):
        output = folder/'adapter'
        if (output/'training.json').exists() and json.loads((output/'training.json').read_text())['status']=='completed':
            self.save(phase='evaluate')
            return
        command = self.training_command(folder)
        if output.exists() and any(output.iterdir()):
            args = training_parser().parse_args(command[2:])
            binding = checkpoint_binding(args)
            if list(output.glob('checkpoint-*/complete.json')):
                latest_checkpoint(output, binding)  # Refuse a corrupt or unbound resume.
                command.append('--resume')
            else:
                if json.loads((output/'run.json').read_text()) != binding:
                    raise ValueError('Incomplete output belongs to different training settings')
                output.rename(folder/f'adapter-incomplete-{time.time_ns()}')
        pause = self.path/'pause-training'
        pause.unlink(missing_ok=True)
        self.spawn(command, folder/'training.log')
        while self.child.poll() is None:
            self.save(detail='CUDA adapter training')
            if self.interrupted():
                pause.touch()
                self.save(detail='Saving checkpoint before pause or stop')
            progress = output/'progress.json'
            if progress.exists():
                self.save(training=json.loads(progress.read_text()))
            time.sleep(1)
        code = self.child.returncode
        self.stop_child()
        if code:
            raise RuntimeError('Training failed; see '+str(folder/'training.log'))
        metadata = json.loads((output/'training.json').read_text())
        if metadata['status'] == 'completed':
            self.save(phase='evaluate')

    def evaluate(self, folder):
        from evaluation import request
        from recursive_agent import solve, grade
        output = folder/'adapter'
        binding = json.loads((output/'run.json').read_text())
        checkpoints = sorted((p.parent for p in output.glob('checkpoint-*/complete.json')
                              if p.parent.name[11:].isdigit()), key=lambda p:int(p.name[11:]))
        if not checkpoints:
            raise ValueError('No complete candidate checkpoint exists')
        tasks = development_tasks(self.path/'development-tasks.json') if self.state.get('development_tasks_sha256') else development_tasks()
        previous = self.state.get('best_dev')
        best_candidate = None
        for checkpoint in checkpoints:
            verified_checkpoint(checkpoint, binding)
            command = [str(ROOT/'.cache/train-env/bin/python'), str(ROOT/'research/adapter_eval_server.py'),
                       '--adapter', str(checkpoint)]
            if not self.server(command, 8090, folder/f'evaluation-{checkpoint.name}.log'):
                return
            reports = {}
            for mode in ('base', 'adapter'):
                suffix = '-'+checkpoint.name if mode=='adapter' else ''
                path = self.path/'dev-base.json' if mode=='base' else folder/f'dev-{mode}{suffix}.json'
                report = json.loads(path.read_text()) if path.exists() else {'tasks': [], 'split': 'dev',
                    'tasks_sha256': sha256(ROOT/'research/rlm_tasks.json'), 'settings': SETTINGS, 'limits': LIMITS,
                    'development_tasks_sha256': self.state.get('development_tasks_sha256'),
                    'adapter_sha256': sha256(checkpoint/'adapter_model.safetensors') if mode=='adapter' else None}
                request('http://127.0.0.1:8090', '/mode', {'mode': mode})
                for task in tasks[len(report['tasks']):]:
                    research_best = (self.state.get('research_score', 0), self.state.get('research_regressions', 0)) if self.state.get('research') else None
                    if (mode=='adapter' and self.state.get('fast_reject')
                            and cannot_improve(report['tasks'], reports['base'], previous, research_best)):
                        report['early_rejected'] = True
                        report['rejection_reason'] = 'No completion can meet promotion or research retention rules'
                        atomic_json(path, report)
                        break
                    self.save(detail=f'Development check {checkpoint.name} {mode}: '+task['id'],
                              evaluation={'checkpoint':checkpoint.name, 'mode':mode,
                                          'completed':len(report['tasks']), 'total':len(tasks)})
                    if self.interrupted():
                        return
                    row = solve(task, 'http://127.0.0.1:8090', **SETTINGS, **LIMITS)
                    if row.get('patch'):
                        row.update(grade(task, row['patch']))
                    report['tasks'].append(row)
                    atomic_json(path, report)
                    self.save(evaluation={'checkpoint':checkpoint.name, 'mode':mode,
                                          'completed':len(report['tasks']), 'total':len(tasks)})
                reports[mode] = report
            keep = accepted(reports['adapter'], reports['base'], previous, expected=len(tasks))
            passed = sum(r.get('passed') is True for r in reports['adapter']['tasks'])
            prior = {r['id']:r.get('passed') is True for r in (previous or reports['base'])['tasks']}
            losses = sum(prior[r['id']] and not r.get('passed') for r in reports['adapter']['tasks'])
            complete = len(reports['adapter']['tasks'])==len(tasks)
            candidate = {'passed':passed, 'regressions':losses, 'total':len(tasks), 'checkpoint':checkpoint.name,
                         'evaluated':len(reports['adapter']['tasks']), 'complete':complete,
                         'early_rejected':reports['adapter'].get('early_rejected',False)}
            if best_candidate is None or (complete,passed,-losses) > (best_candidate['complete'],best_candidate['passed'],-best_candidate['regressions']):
                best_candidate = candidate
            if (self.state.get('research') and len(reports['adapter']['tasks'])==len(tasks)
                    and {r['id'] for r in reports['adapter']['tasks']}==prior.keys() and losses<=1):
                old = (self.state.get('research_score',sum(prior.values())), -self.state.get('research_regressions',0))
                if (passed,-losses)>old:
                    # Research may explore one lost task; this never relaxes the promotion rule.
                    self.state.update(research_adapter=str(checkpoint), research_score=passed, research_regressions=losses)
            self.save(candidate_best=best_candidate)
            self.stop_child()
            if keep:
                break
        result = {'round': self.state['round'], 'base_passed': sum(r['passed'] for r in reports['base']['tasks']),
                  'adapter_passed': sum(r['passed'] for r in reports['adapter']['tasks']), 'total': len(tasks),
                  'accepted': keep, 'checkpoint': checkpoint.name, 'best_candidate': best_candidate,
                  'evaluated':len(reports['adapter']['tasks']), 'complete':complete,
                  'early_rejected':reports['adapter'].get('early_rejected',False),
                  'warning': 'Small repeated development set; no general quality claim. Holdout unused.'}
        atomic_json(folder/'result.json', result)
        self.state['completed_rounds'].append(result)
        if keep:
            self.state['best_adapter'] = str(checkpoint)
            self.state['best_dev'] = {'tasks': [{'id': r['id'], 'passed': r['passed']}
                                               for r in reports['adapter']['tasks']]}
            self.state['accepted_rounds'].append(self.state['round'])
            if self.state.get('research'):
                self.state.update(research_adapter=str(checkpoint), research_score=passed, research_regressions=0)
        stop = should_stop_for_quality(self.state['completed_rounds'], self.state.get('research',False))
        self.save(round=self.state['round']+1, phase='collect',
                  status='completed' if stop else 'running', collected=0, passed=0, training=None, evaluation=None,
                  detail='Stopped after three rounds without development improvement'
                  if stop else 'Starting a new data collection round')

    def run(self):
        if self.state['status'] in ('completed', 'stopped'):
            raise ValueError('This session is finished. Choose a new --session folder to start another.')
        self.save(status='running')
        try:
            while True:
                self.save()
                if self.state['status']=='completed':
                    return
                if self.command()=='stop':
                    self.save(status='stopped', detail='Stopped with saved progress')
                    return
                if self.state['active_seconds'] >= self.state['limit_seconds']:
                    self.save(status='completed', detail='Active time limit reached; progress retained')
                    return
                if self.command()=='pause':
                    self.stop_child()
                    self.save(status='paused', detail='GPU released; checkpoints and task progress retained')
                    time.sleep(1)
                    continue
                self.save(status='running')
                if self.state['round'] > 64:
                    self.save(status='completed', detail='64-round ceiling reached')
                    return
                if os.statvfs(self.path).f_bavail*os.statvfs(self.path).f_frsize < 10*1024**3:
                    raise ValueError('Less than 10 GiB free disk; session stopped before training')
                if self.state['phase']=='collect' and sum(p.stat().st_size for p in self.path.rglob('*') if p.is_file()) > 20*1024**3:
                    self.save(status='completed', detail='20 GiB session storage ceiling reached')
                    return
                folder = self.path/f"round-{self.state['round']:03d}"
                folder.mkdir(exist_ok=True)
                phase = self.state['phase']
                getattr(self, {'collect':'collect', 'export':'dataset', 'train':'train', 'evaluate':'evaluate'}[phase])(folder)
                self.stop_child()
        except BaseException as error:
            self.save(status='failed', detail=f'{type(error).__name__}: {error}')
            raise
        finally:
            self.stop_child()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run','pause','resume','stop','status'))
    parser.add_argument('--session', type=Path, default=DEFAULT)
    parser.add_argument('--hours', type=float, default=12)
    parser.add_argument('--research', action='store_true', help='Explore short anchored updates without the three-round plateau stop; promotion stays strict')
    parser.add_argument('--fast-reject', action='store_true', help='Stop checking a candidate once improvement is impossible; promotion still requires all tasks')
    parser.add_argument('--model', help='Teacher GGUF path, Windows or Linux; required for a new session')
    args = parser.parse_args()
    if not .01 <= args.hours <= 24:
        parser.error('Use a session limit between 0.01 and 24 hours')
    try:
        if args.action == 'status':
            print((args.session/'status.json').read_text() if (args.session/'status.json').exists() else '{}')
        elif args.action != 'run':
            args.session.mkdir(parents=True, exist_ok=True)
            atomic_json(args.session/'command.json', {'action':args.action, 'requested_at':time.time()})
            print(args.action+' requested')
        else:
            model = args.model
            if model and PureWindowsPath(model).drive:
                model = subprocess.check_output(['wslpath', '-u', model], text=True).strip()
            if not model and (args.session/'status.json').exists():
                model = json.loads((args.session/'status.json').read_text())['teacher_model']
            if not model or not Path(model).is_file():
                raise ValueError('Teacher GGUF file is missing; pass --model')
            Session(args.session, args.hours, Path(model), research=args.research, fast_reject=args.fast_reject).run()
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, str(error)+'\n')


if __name__ == '__main__':
    main()
