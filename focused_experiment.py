"""Finite local research with bound inputs, saved stages and durable Windows controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import uuid

from train_adapter import atomic_json
from training_data import sha256

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT/'.cache/learning/qwen35-go-ts'
TERMINAL = ('completed', 'stopped', 'budget_exhausted')


def read(path, fallback=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else fallback


def locked(path):
    if not Path(path).exists():
        return False
    with Path(path).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def owned_group(owner, session):
    """A surviving child retains the token even when its group leader exited."""
    if not owner:
        return False
    wanted = ('LCA_FOCUSED_TOKEN='+owner['token']).encode()
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            stat = (proc/'stat').read_text().split(') ', 1)[1].split()
            if int(stat[2]) != owner['pid']:
                continue
            env = (proc/'environ').read_bytes().split(b'\0')
            if wanted in env and ('LCA_FOCUSED_OWNER='+str(session)).encode() in env:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def stop_owned_group(owner, session, seconds=30):
    if not owned_group(owner, session):
        return
    try:
        os.killpg(owner['pid'], signal.SIGTERM)
        deadline = time.monotonic()+seconds
        while owned_group(owner, session) and time.monotonic() < deadline:
            time.sleep(.2)
        if owned_group(owner, session):
            os.killpg(owner['pid'], signal.SIGKILL)
    except ProcessLookupError:
        pass


class Experiment:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.config = read(self.path/'config.json')
        self.state = read(self.path/'status.json')
        if not self.config or not self.state:
            raise ValueError('Prepare this experiment before starting it')
        self.exiting = False

    @classmethod
    def create(cls, path, config):
        path = Path(path).resolve()
        if (path/'status.json').exists():
            raise ValueError('Experiment already exists; use Resume')
        stages = config.get('stages', [])
        if not stages or len({s['name'] for s in stages}) != len(stages):
            raise ValueError('Use a fixed sequence of unique stages')
        for stage in stages:
            result = Path(stage['result'])
            if result.is_absolute() or '..' in result.parts:
                raise ValueError('Stage result path must stay inside the session')
            if not isinstance(stage['command'], list) or not stage['command'] or any(
                    not isinstance(v, str) for v in stage['command']):
                raise ValueError('Stage command must contain explicit arguments')
        if not 0 < config.get('limit_seconds', 21600) <= 21600:
            raise ValueError('Training budget must be positive and at most six hours')
        path.mkdir(parents=True, exist_ok=True)
        atomic_json(path/'config.json', config)
        atomic_json(path/'command.json', {'action': 'pause', 'requested_at': time.time()})
        atomic_json(path/'status.json', {'schema': 1, 'focused': True, 'status': 'paused',
            'phase': stages[0]['name'], 'stage_index': 0, 'round': 1, 'active_seconds': 0,
            'training_seconds': 0, 'limit_seconds': config.get('limit_seconds', 21600),
            'config_sha256': sha256(path/'config.json'), 'completed_stages': [], 'failures': 0,
            'detail': 'Prepared finite comparison; Resume starts the first saved stage', 'updated_at': time.time()})
        result = cls(path)
        result.validate()
        return result

    def save(self, **changes):
        self.state.update(changes, updated_at=time.time())
        atomic_json(self.path/'status.json', self.state)

    def desired(self):
        return read(self.path/'command.json', {'action': 'pause'})['action']

    def control(self, action):
        if action not in ('pause', 'resume', 'stop'):
            raise ValueError('Use Pause, Resume or Stop')
        if action == 'resume' and self.state['status'] in TERMINAL:
            raise ValueError('This finite experiment is finished; its clock and results stay preserved')
        atomic_json(self.path/'command.json', {'action': action, 'requested_at': time.time()})

    def snapshot(self):
        value = dict(self.state, desired=self.desired(), worker_alive=locked(self.path/'worker.lock'))
        if not value['worker_alive'] and value['status'] not in TERMINAL:
            if owned_group(read(self.path/'process.json'), self.path):
                value.update(status='interrupted', detail='A saved model process still needs cleanup; GPU release is not confirmed')
            elif value['desired'] in ('pause', 'stop'):
                value['status'] = 'paused' if value['desired'] == 'pause' else 'stopped'
            elif value['status'] != 'failed':
                value.update(status='interrupted', detail='Worker is not running; saved work is ready for Resume')
        if value['stage_index'] < len(self.config['stages']):
            stage = self.config['stages'][value['stage_index']]
            progress = read(self.path/stage.get('progress', 'progress.json'), {})
            value['stage_progress'] = progress
            if stage.get('train'):
                value['training'] = progress
        value['stage_total'] = len(self.config['stages'])
        return value

    def validate(self):
        if sha256(self.path/'config.json') != self.state['config_sha256']:
            raise ValueError('Experiment settings changed; prepare a separate experiment')
        for name, digest in self.config['inputs'].items():
            if not Path(name).is_file() or sha256(name) != digest:
                raise ValueError('Bound input changed: '+name)
        for result in self.state['completed_stages']:
            if sha256(self.path/result['result']) != result['sha256']:
                raise ValueError('Completed result changed: '+result['name'])

    def finish_stage(self):
        stage = self.config['stages'][self.state['stage_index']]
        result = read(self.path/stage['result'], {})
        if result.get('completed') is not True and result.get('status') != 'completed':
            raise ValueError('Stage result is incomplete: '+stage['name'])
        done = self.state['completed_stages']+[{'name': stage['name'], 'result': stage['result'],
            'sha256': sha256(self.path/stage['result'])}]
        self.save(stage_index=self.state['stage_index']+1, completed_stages=done, failures=0)

    def recover_child(self):
        owner = read(self.path/'process.json')
        if not owner:
            return
        stop_owned_group(owner, self.path)
        (self.path/'process.json').unlink(missing_ok=True)

    def execute_stage(self, stage):
        pause = self.path/'pause.requested'
        pause.unlink(missing_ok=True)
        gpu = None
        process = None
        owner = None
        clock = time.monotonic()
        interruption = None
        try:
            if stage.get('gpu', True):
                lock = ROOT/'.cache/learning/gpu.lock'
                lock.parent.mkdir(parents=True, exist_ok=True)
                gpu = lock.open('a')
                fcntl.flock(gpu, fcntl.LOCK_EX|fcntl.LOCK_NB)
                for port in (8080, 8090):
                    with socket.socket() as probe:
                        if probe.connect_ex(('127.0.0.1', port)) == 0:
                            raise RuntimeError('Close the model server before starting the experiment')
            self.save(status='running', phase=stage['name'], detail='Running '+stage['name'],
                      round=self.state['stage_index']+1)
            log_path = self.path/(stage['name']+'.log')
            with log_path.open('ab') as log:
                offset = log.tell()
                command = list(stage['command'])
                if stage.get('train') and (self.path/Path(stage['result']).parent/'request.json').exists():
                    command.append('--resume')
                token = uuid.uuid4().hex
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1',
                    'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                    'LCA_FOCUSED_OWNER': str(self.path), 'LCA_FOCUSED_TOKEN': token,
                    'UNSLOTH_COMPILE_LOCATION': str(ROOT/'.cache/unsloth-compiled'), 'PYTHONPATH': str(ROOT)})
            owner = {'pid': process.pid, 'token': token}
            atomic_json(self.path/'process.json', owner)
            while process.poll() is None:
                now = time.monotonic(); elapsed = now-clock; clock = now
                budget = self.state['training_seconds']+elapsed if stage.get('train') else self.state['training_seconds']
                self.save(active_seconds=self.state['active_seconds']+elapsed, training_seconds=budget)
                requested = self.desired() != 'resume' or self.exiting or (stage.get('train') and budget >= self.state['limit_seconds'])
                if requested and interruption is None:
                    interruption = now
                    self.save(status='pausing' if self.desired() != 'stop' else 'stopping',
                              detail='Saving progress and releasing the GPU')
                    pause.touch()
                    if not stage.get('train'):
                        os.killpg(process.pid, signal.SIGTERM)
                if interruption and now-interruption > (180 if stage.get('train') else 30):
                    os.killpg(process.pid, signal.SIGKILL)
                time.sleep(1)
            elapsed = time.monotonic()-clock
            self.save(active_seconds=self.state['active_seconds']+elapsed,
                      training_seconds=self.state['training_seconds']+(elapsed if stage.get('train') else 0))
            if interruption:
                return False
            if process.returncode:
                with log_path.open('rb') as log:
                    log.seek(max(offset, log_path.stat().st_size-6000))
                    tail = log.read().decode('utf-8', errors='replace')
                raise RuntimeError(f'{stage["name"]} exited {process.returncode}: {tail}')
            self.finish_stage()
            return True
        finally:
            stop_owned_group(owner, self.path)
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
            (self.path/'process.json').unlink(missing_ok=True)
            if gpu:
                gpu.close()

    def run(self):
        def exit_requested(*_):
            self.exiting = True
        signal.signal(signal.SIGTERM, exit_requested)
        signal.signal(signal.SIGINT, exit_requested)
        with (self.path/'worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.recover_child(); self.validate()
            last_command = self.state.get('last_command', 0)
            while self.state['status'] not in TERMINAL:
                command = read(self.path/'command.json')
                if command['requested_at'] > last_command:
                    last_command = command['requested_at']
                    if command['action'] == 'resume':
                        self.save(status='running', failures=0, last_command=last_command)
                    else:
                        self.save(last_command=last_command)
                if self.desired() == 'stop':
                    self.save(status='stopped', detail='Stopped; checkpoints and reports remain saved'); return
                if self.exiting:
                    self.save(status='interrupted', detail='Worker closed; saved command controls the next login'); return
                if self.desired() == 'pause':
                    self.save(status='paused', detail='GPU released. Resume continues saved work')
                    time.sleep(2); continue
                if self.state['status'] == 'failed':
                    time.sleep(2); continue
                if self.state['stage_index'] == len(self.config['stages']):
                    self.save(status='completed', phase='completed', detail='Finite experiment and scheduled comparisons completed')
                    return
                if self.config['stages'][self.state['stage_index']].get('train') and self.state['training_seconds'] >= self.state['limit_seconds']:
                    self.save(status='budget_exhausted', detail='Six active training hours reached; checkpoint retained, experiment incomplete')
                    return
                try:
                    self.validate()
                    self.execute_stage(self.config['stages'][self.state['stage_index']])
                except (OSError, ValueError, RuntimeError) as error:
                    count = self.state['failures']+1
                    self.save(status='failed' if count >= 3 or isinstance(error, ValueError) else 'waiting',
                              failures=count, detail=str(error)[-6000:])
                    if self.state['status'] == 'waiting':
                        for _ in range(60):
                            if self.desired() != 'resume' or self.exiting:
                                break
                            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'run', 'status', 'pause', 'resume', 'stop'))
    parser.add_argument('--session', type=Path, default=DEFAULT)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    try:
        if args.action == 'init':
            if not args.config:
                raise ValueError('Pass a reviewed experiment configuration')
            experiment = Experiment.create(args.session, read(args.config))
        else:
            experiment = Experiment(args.session)
            if args.action == 'run':
                experiment.run(); return
            if args.action != 'status':
                experiment.control(args.action)
        print(json.dumps(experiment.snapshot()))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, str(error)+'\n')


if __name__ == '__main__':
    main()
