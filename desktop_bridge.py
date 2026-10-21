"""Private JSON interface for Training Studio; no network listener or shell commands."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from focused_experiment import Experiment, open_experiment, read
from focused_report import compare
from qwen35_train import atomic_json, load_verified, sha256

ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / '.cache/learning'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def session_path(name):
    if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', name):
        raise ValueError('Use a session name with letters, numbers, dashes or underscores')
    path = (SESSIONS / name).resolve()
    if path.parent != SESSIONS.resolve():
        raise ValueError('Session must stay in the private learning folder')
    return path


def tail(path):
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 24000))
        text = stream.read(24000).decode('utf-8', errors='replace')
    return re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)


def checked_report(path):
    state = json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {}
    if state.get('continuous'):
        name = state.get('last_report')
        if not name: return None
        file = (path/name).resolve()
        if not file.is_relative_to(path.resolve()) or sha256(file) != state.get('last_report_sha256'):
            raise ValueError('Retained loop report changed')
    else:
        file = path / 'comparison.json'
    if not file.exists():
        return None
    report = json.loads(file.read_text())
    state_file = path / 'status.json'
    if state_file.exists() and not state.get('continuous'):
        state = json.loads(state_file.read_text())
        saved = next((s for s in state.get('completed_stages', []) if s.get('result') == 'comparison.json'), None)
        if saved is None:
            return None
        if saved['sha256'] != sha256(file):
            raise ValueError('Completed comparison changed; result is not trusted')
    if report.get('completed') is not True:
        return None
    gate = report.get('pilot_gate')
    if gate and gate.get('completed') is True and 'comparisons' not in report:
        report = dict(report, inputs=gate.get('inputs'), reason=gate.get('reason'),
            comparisons=dict(training_effect=dict(repairs=dict(python=dict(
                before=gate['before'], after=gate['after'], total=gate['total'],
                gained=gate['gained_families'], lost=gate['lost'])))))
    if not report.get('inputs'):
        raise ValueError('Comparison has no bound evidence')
    for name, digest in report['inputs'].items():
        item = (path / name).resolve()
        if not item.is_relative_to(path.resolve()):
            # Reused weights must match the exact external file bound before the run.
            config = path/'config.json'
            if (not config.is_file() or sha256(config) != state.get('config_sha256')
                    or read(config).get('inputs', {}).get(str(item)) != digest):
                raise ValueError('Comparison external evidence is not bound to this session')
        if not item.is_file() or sha256(item) != digest:
            raise ValueError('Comparison evidence changed; result is not trusted')
    return report


def verdict(report):
    if not report or report.get('completed') is not True:
        return {'label': 'Not yet measured', 'tone': 'neutral', 'before': 0, 'after': 0,
                'total': 0, 'gained': 0, 'lost': 0, 'languages': []}
    suites = report['comparisons']['training_effect']
    rows = [dict(suite=suite, language=language, **value)
            for suite, languages in suites.items() for language, value in languages.items()]
    before, after, total = (sum(row[key] for row in rows) for key in ('before', 'after', 'total'))
    gained = sum(len(row['gained']) for row in rows)
    lost = sum(len(row['lost']) for row in rows)
    label, tone = ('Regressed on these tests', 'bad') if after < before else (
        ('Mixed results', 'mixed') if lost else (
        ('Improved on these tests', 'good') if gained else ('No measured change', 'neutral')))
    return dict(label=label, tone=tone, before=before, after=after, total=total,
                gained=gained, lost=lost, languages=rows)


def task_preview(path, state):
    if not state.get('continuous'):
        return None
    folder=path/f"round-{int(state.get('cycle',1)):06d}"
    identity=state.get('stage_progress',{}).get('task')
    tasks=read(folder/'tasks.json',{})
    task=next((t for t in tasks.get('train',[])+tasks.get('validation',[]) if t['id']==identity),None)
    if not task or task.get('filename') not in task.get('editable',[]):
        return None
    code=task.get('files',{}).get(task['filename'],'')
    phase=state.get('phase','').split('/')[-1]
    if phase in ('collect','baseline','candidate'):
        attempt=read(folder/(phase+'-attempt.json'),{})
        if attempt.get('id')==identity:
            code=attempt.get('answer',code)
    return dict(filename=task['filename'],language=task['language'],code=code[:12000],truncated=len(code)>12000)


def agent_curriculum_status(path, state, config):
    """Read pilot evidence for display; never score partial results or control work."""
    if state.get('phase') == 'prepare':
        state['readiness'] = 'Preparing and verifying training functions. Quality is not yet measured.'
        return
    if state.get('phase') == 'collect':
        state['readiness'] = 'Collecting executed student attempts and verified corrections. Quality is not yet measured.'
        return

    def local_file(name):
        file = (path/name).resolve()
        if not file.is_relative_to(path.resolve()):
            raise ValueError('Pilot display evidence must stay in its session')
        return file

    bindings = dict(config.get('inputs', {}))
    collected = next((stage for stage in state.get('completed_stages', [])
                      if stage.get('result') == 'curriculum/feasibility.json'), None)
    if collected:
        evidence = local_file('curriculum/feasibility.json')
        if not evidence.is_file() or sha256(evidence) != collected.get('sha256'):
            raise ValueError('Pilot display evidence changed: collection result')
        saved = read(evidence)
        for name, digest in [('curriculum/feasibility.json', collected['sha256']),
                             ('curriculum/train.jsonl', saved.get('dataset_sha256')),
                             ('curriculum/train.jsonl.manifest.json', saved.get('manifest_sha256'))]:
            local_file(name)
            bindings[str(path/name)] = digest

    def frozen_file(name):
        # A pilot may reuse an immutable curriculum through an explicitly bound symlink.
        file = path/name
        if not file.is_file() or sha256(file) != bindings.get(str(file)):
            raise ValueError('Pilot display evidence changed: '+name)
        return file

    dataset = frozen_file('curriculum/train.jsonl')
    manifest = frozen_file('curriculum/train.jsonl.manifest.json')
    preflight = read(frozen_file('curriculum/feasibility.json'))
    rows, metadata = load_verified(dataset, manifest)
    if (metadata['verification'] != 'local_execution'
            or any(preflight.get(key) is not True for key in
                   ('passed', 'all_agent_actions_included', 'complete_trajectories'))
            or preflight.get('dataset_sha256') != metadata['dataset_sha256']
            or preflight.get('manifest_sha256') != sha256(manifest)):
        raise ValueError('Pilot lessons require intact execution and trajectory evidence')
    state['training_examples'] = len(rows)
    state['repl_examples'] = sum(row.get('example_kind') == 'executed_repl_repair' for row in rows)
    kind = 'corrective' if metadata.get('binding', {}).get('construction') == 'student_state_corrective_imitation' else 'scripted'
    state['readiness'] = (f"{state['repl_examples']} verified {kind} REPL training rows; "
                          f"{len(rows)} training rows total. Waiting for a complete comparison.")

    index = state['stage_index']
    if index >= len(config['stages']):
        return
    stage = config['stages'][index]
    result = local_file(stage.get('result', ''))
    suite = result.stem
    if stage.get('train') or suite not in ('development', 'holdout'):
        return
    evaluation_file = frozen_file('evaluation.json')
    evaluation = read(evaluation_file)
    expected = [(entry['condition'], entry['task']['id']) for entry in evaluation[suite]]
    anchors = [task['id'] for task in evaluation['retention']] if suite == 'development' else []
    saved = read(result, None)
    completed, retention = [], []
    if saved is not None:
        binding = saved.get('binding', {})
        completed, retention = saved.get('rows', []), saved.get('retention', [])
        if (binding.get('evaluation_sha256') != sha256(evaluation_file) or binding.get('suite') != suite
                or not isinstance(completed, list) or not isinstance(retention, list)
                or any(not isinstance(row, dict) or type(row.get('passed')) is not bool
                       or row.get('infrastructure_errors') for row in completed+retention)
                or [(row.get('condition'), row.get('id')) for row in completed] != expected[:len(completed)]
                or [row.get('id') for row in retention] != anchors[:len(retention)]
                or (retention and len(completed) != len(expected))):
            raise ValueError('Pilot progress does not match its frozen comparison tasks')
    current = read(local_file(result.parent/'task.json'), {})
    active = (len(completed) < len(expected) and
              (current.get('condition'), current.get('task', {}).get('id')) == expected[len(completed)])
    if saved is None and not active:
        return  # No task has started yet; preserve the real model-loading status.
    state['task_preview'] = None
    progress = dict(phase='repl-baseline' if result.parent.name == 'baseline' else 'repl-candidate',
                    completed=len(completed)+len(retention), total=len(expected)+len(anchors))
    if active:
        # Use frozen public source, not mutable task contents or private checks/answers.
        task = evaluation[suite][len(completed)]['task']
        progress['task'] = task['id']
        filename = next((name for name in task.get('editable', []) if name in task.get('files', {})), None)
        if filename:
            code = task['files'][filename]
            state['task_preview'] = dict(filename=filename, language=task.get('language', 'python'),
                                         code=code[:12000], truncated=len(code)>12000)
    elif len(completed) == len(expected):
        progress['task'] = 'Retention checks' if len(retention) < len(anchors) else 'Comparison tasks completed'
    state['stage_progress'] = progress


def snapshot(name):
    path = session_path(name)
    experiment = open_experiment(path)
    value = experiment.snapshot()
    value['session'] = name
    value['task_preview'] = task_preview(path,value)
    if experiment.config.get('learning_method') == 'agent_curriculum':
        agent_curriculum_status(path, value, experiment.config)
    report = checked_report(path)
    if experiment.config.get('learning_method') == 'agent_curriculum' and report is not None:
        value['readiness'] = report.get('reason') or 'Comparison completed.'
    value['quality'] = verdict(report)
    value['report'] = report
    phase = value.get('phase', '')
    # Only read the log of a configured stage, never a caller-supplied path.
    log = path / (phase + '.log')
    value['log'] = tail(log) if phase in [s['name'] for s in experiment.config['stages']] and log.resolve().is_relative_to(path.resolve()) else ''
    value['checkpoints'] = []
    adapters = [path/Path(s['result']).parent for s in experiment.config['stages'] if s.get('train')] or [path/'adapter']
    markers = path.glob('round-*/adapter/checkpoint-*/complete.json') if value.get('continuous') else (
        marker for adapter in adapters if adapter.resolve().is_relative_to(path.resolve())
        for marker in adapter.glob('**/complete.json'))
    for marker in sorted(markers)[-12:]:
        if marker.is_symlink() or not marker.resolve().is_relative_to(path.resolve()):
            continue
        value['checkpoints'].append({'path': str(marker.parent.relative_to(path)),
                                    'saved_at': marker.stat().st_mtime})
    value['stages'] = [{'name': s['name'], 'done': i < value['stage_index']}
                       for i, s in enumerate(experiment.config['stages'])]
    value['gpu'] = None
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4, check=True)
        fields = result.stdout.strip().splitlines()[0].split(', ')
        value['gpu'] = dict(zip(('name','used_mb','total_mb','utilization','watts'), fields))
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    value['observed_at'] = time.time()
    return value


def dataset_info(dataset):
    file = Path(dataset).expanduser().resolve()
    if not file.is_file() or file.stat().st_size > 512 * 1024 * 1024:
        raise ValueError('Choose an existing JSONL file smaller than 512 MiB')
    rows, metadata = load_verified(file, Path(str(file) + '.manifest.json'))
    if len(rows) < 2000 or any(sum(r['language'] == l for r in rows) < 250 for l in ('go','typescript')):
        raise ValueError('This recipe requires at least 2,000 verified examples, with at least 250 per language')
    return {'rows': len(rows), 'languages': dict(Counter(r['language'] for r in rows)),
            'verification': metadata['verification'], 'dataset_sha256': sha256(file),
            'preview': '\n\n'.join(m['role'] + ': ' + m['content'][:700] for m in rows[0]['messages'][-2:])}


def prepare(request):
    path = session_path(request['name'])
    hours = float(request.get('hours', 6))
    if not 0 < hours <= 6:
        raise ValueError('Choose between 0 and 6 active training hours')
    if path.exists():
        raise ValueError('That experiment already exists. Choose a new name or Resume it')
    dataset = Path(request['dataset']).expanduser().resolve()
    info = dataset_info(dataset)
    python = ROOT / '.cache/rlm-env/bin/python'
    trainer = ROOT / '.cache/qwen35-env/bin/python'
    for required in (python, trainer, ROOT / '.cache/polyglot-runtime.json'):
        if not required.is_file():
            raise ValueError('Prepare the Qwen3.5 CUDA backend and benchmark runtime first: ' + str(required))
    # Keep the source and private data immutable for the duration of an experiment.
    sources = [ROOT / name for name in ('focused_experiment.py','focused_evaluation.py',
        'qwen35_train.py','qwen35_server.py','structured_output.py','desktop_bridge.py','focused_report.py','focused_data.py','focused_replay.py',
        'train_adapter.py','training_data.py','evaluation.py','launcher.py','polyglot_runtime.py',
        'research/polyglot_benchmark.py','research/adapter_eval_server.py')]
    inputs = {str(p): sha256(p) for p in sources}
    path.mkdir(parents=True)
    try:
        data = path / 'data'; data.mkdir()
        target = data / 'train.jsonl'
        shutil.copyfile(dataset, target)
        manifest = Path(str(target) + '.manifest.json')
        shutil.copyfile(str(dataset) + '.manifest.json', manifest)
        dataset_info(target)
        inputs.update({str(p): sha256(p) for p in (target, manifest, ROOT / '.cache/polyglot-runtime.json')})
        stages = []
        def audit(mode, label):
            return {'name': label, 'command': [str(python), str(ROOT/'focused_evaluation.py'),
                '--engine','qwen35','--mode',mode,'--suite','audit','--batch-size','2',
                '--output',str(path/label)] + (['--adapter',str(path/'adapter')] if mode=='adapter' else []),
                'result':label+'/summary.json','progress':label+'/progress.json','gpu':True}
        stages.append(audit('base','qwen35-base-audit'))
        stages.append({'name':'train','command':[str(trainer),str(ROOT/'qwen35_train.py'),
            '--dataset',str(target),'--manifest',str(manifest),'--output',str(path/'adapter'),
            '--model','Qwen/Qwen3.5-4B','--revision',REVISION,'--max-steps','1000','--max-epochs','1',
            '--max-length','4096','--rank','8','--gradient-accumulation','8','--learning-rate','0.00005',
            '--save-steps','10','--min-examples','2000','--min-per-language','250',
            '--pause-file',str(path/'pause.requested')], 'result':'adapter/training.json',
            'progress':'adapter/progress.json','gpu':True,'train':True})
        stages.append(audit('adapter','qwen35-trained-audit'))
        stages.append({'name':'comparison','command':[str(python),str(ROOT/'desktop_bridge.py'),
            '--summarize',request['name']], 'result':'comparison.json','gpu':False})
        Experiment.create(path, {'schema':1,'purpose':'One Qwen3.5 trial with matched coding benchmarks',
            'limit_seconds':hours*3600,'inputs':inputs,'stages':stages})
        return {'session': request['name'], 'dataset': info, 'status':'paused'}
    except Exception:
        # Only remove this newly created directory; never existing user experiments.
        shutil.rmtree(path)
        raise


def summarize(name):
    path = session_path(name)
    loaded, inputs = {}, {}
    for model in ('qwen35-base', 'qwen35-trained'):
        folder = path / (model + '-audit')
        summary = json.loads((folder/'summary.json').read_text())
        if summary.get('completed') is not True or summary['binding'].get('smoke_limit'):
            raise ValueError('Both complete benchmarks are required')
        rows = {l: json.loads((folder/(l+'.json')).read_text()) for l in summary['languages']}
        checks = compare(rows, rows)
        for l, value in checks.items():
            if value['after'] != summary['languages'][l]['passed'] or value['total'] != summary['languages'][l]['total']:
                raise ValueError('Language summary differs from the test results')
        if sum(v['after'] for v in checks.values()) != summary['passed'] or sum(v['total'] for v in checks.values()) != summary['total']:
            raise ValueError('Summary differs from the test results')
        for file in [folder/'summary.json'] + [folder/(l+'.json') for l in rows]:
            inputs[str(file.relative_to(path))] = sha256(file)
        loaded[model] = (summary, rows)
    a, b = loaded.values()
    # Mode and adapter path are the intended change; all generation conditions must match.
    keys = ('datasets','max_output_tokens','temperature','thinking','batch_size','prompt',
            'container_image','sources','engine','model','revision','precision','runtime')
    if any(k not in a[0]['binding'] or k not in b[0]['binding'] or a[0]['binding'][k] != b[0]['binding'][k] for k in keys):
        raise ValueError('Benchmark conditions differ')
    report = {'completed':True,'automatically_promoted':False,'inputs':inputs,
        'models':{k:{'audit':{field:v[0][field] for field in ('passed','total','languages','macro_pass_at_1')}} for k,v in loaded.items()},
        'comparisons':{'training_effect':{'audit':compare(a[1], b[1])}},
        'limitations':['Public benchmarks may overlap model pretraining. Results do not prove general coding quality.',
                       'Data provenance is checked; publisher verification is not a local replay.'],
        'decision':'Review gained and lost tasks before choosing an adapter.'}
    atomic_json(path/'comparison.json',report)
    return report


def dispatch(request):
    action = request.get('action')
    if action == 'list':
        sessions, history = [], []
        for file in sorted(SESSIONS.glob('*/status.json'), key=lambda p:p.stat().st_mtime, reverse=True):
            try:
                state=json.loads(file.read_text())
                if state.get('focused') and (file.parent/'config.json').is_file():
                    sessions.append({'name':file.parent.name,'status':state['status']})
                    if (file.parent/'comparison.json').is_file() or state.get('last_report'):
                        quality = verdict(checked_report(file.parent))
                        if quality['total']:
                            history.append({'session':file.parent.name,'updated_at':state.get('updated_at',0),**quality})
            except (ValueError, OSError, KeyError):
                continue
        return {'sessions':sessions, 'history':history}
    if action == 'validate-data': return dataset_info(request['dataset'])
    if action == 'prepare': return prepare(request)
    if action == 'prepare-loop':
        from coding_loop import CodingLoop
        teacher = request.get('teacher', True)
        if type(teacher) is not bool: raise ValueError('Teacher selection must be true or false')
        loop = CodingLoop.prepare(session_path(request['name']), teacher=teacher)
        return {'session':request['name'],'status':loop.snapshot()['status']}
    if action not in ('status','pause','resume','stop'):
        raise ValueError('Unsupported desktop action')
    name = request['session']
    if action != 'status': open_experiment(session_path(name)).control(action)
    return snapshot(name)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarize')
    args=parser.parse_args()
    try:
        result = summarize(args.summarize) if args.summarize else dispatch(json.load(sys.stdin))
        print(json.dumps(result))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'error':str(error)})); sys.exit(1)
