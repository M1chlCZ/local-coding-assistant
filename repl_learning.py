"""Executed REPL demonstrations and a separately scored, bounded recursive pilot."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

from focused_experiment import ROOT, read
from qwen35_train import atomic_json, sha256

LIMITS = dict(depth=2, calls=8, output_tokens=6144, seconds=240)


def cleanup_containers(owner):
    result=subprocess.run(['docker','ps','-aq','--filter','label=local-coding-assistant.owner='+owner],
                          capture_output=True,text=True,check=True,timeout=15)
    containers=result.stdout.split()
    if containers:
        subprocess.run(['docker','rm','-f',*containers],capture_output=True,check=True,timeout=30)


def run_child(command, timeout):
    owner=uuid.uuid4().hex
    process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'LCA_REPL_OWNER':owner})
    try:
        try:
            code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError('REPL task exceeded its wall-clock limit') from error
        if code:raise RuntimeError(f'REPL worker exited {code}; inspect the stage log')
    finally:
        # Pause and deadlines both unwind the sandbox before a last-resort forced kill.
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        cleanup_containers(owner)


def mix_rows(rows):
    """Cap each language at twice the smallest pool, alternating lesson kinds."""
    from polyglot_runtime import LANGUAGES
    grouped={l:[r for r in rows if r['language']==l] for l in LANGUAGES}
    cap=2*min(map(len,grouped.values()))
    selected={}
    for language, values in grouped.items():
        kinds={kind:[r for r in values if r.get('example_kind','complete_code')==kind]
               for kind in ('complete_code','correction','executed_repl_repair','teacher_reference')}
        selected[language]=[pool[i] for i in range(max(map(len,kinds.values()),default=0))
                            for pool in kinds.values() if i<len(pool)][:cap]
    return [selected[l][i] for i in range(max(map(len,selected.values()),default=0))
            for l in LANGUAGES if i<len(selected[l])]


def lesson_rows(task, answer):
    from recursive_agent import lesson_prompt
    from training_data import repair_calls
    if task['split'] != 'train' or task.get('kind') == 'teacher_function':
        raise ValueError('REPL lessons require registered training programs')
    original = copy.deepcopy(task)
    if answer.get('initial_answer') and answer.get('first_passed') is False:
        original['files'][task['filename']] = answer['initial_answer']
    prompt = lesson_prompt(original)
    try:
        calls = repair_calls(original, [dict(kind='root', messages=prompt)],
                             {task['filename']: answer['answer']})
    except ValueError as error:
        if str(error) == 'Visible tests do not reproduce the defect':
            return []
        raise
    provenance = dict(dataset=task['provenance']['dataset'], revision=task['provenance']['revision'],
        source_row_sha256=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest())
    return [dict(task_id=task['id']+f':repl-{i}', source_task_id=task['id'], language=task['language'],
        split='train', example_kind='executed_repl_repair',
        messages=messages+[dict(role='assistant',content=response)], provenance=provenance,
        verification=dict(kind='local_execution',passed=True)) for i,(messages,response) in enumerate(calls)]


def build_lessons(jobs, output):
    """Each verified trajectory is durable; interrupted jobs repeat only their own sandbox."""
    binding = hashlib.sha256(json.dumps(jobs,sort_keys=True).encode()).hexdigest()
    saved = read(output, dict(binding=binding, completed=0, rows=[]))
    if saved['binding'] != binding or not 0 <= saved['completed'] <= len(jobs):
        raise ValueError('REPL lesson inputs changed')
    for job in jobs[saved['completed']:]:
        saved['rows'].extend(lesson_rows(job['task'],job['answer']))
        saved['completed'] += 1
        atomic_json(output,saved)
        print(f"REPL lessons: {saved['completed']}/{len(jobs)}",flush=True)
    return saved['rows']


def export_lessons(folder, jobs):
    if not jobs: return []
    path=folder/'repl-jobs.json'
    if path.exists() and read(path) != jobs:
        raise ValueError('Prepared REPL lesson jobs changed')
    atomic_json(path,jobs)
    run_child([str(ROOT/'.cache/rlm-env/bin/python'),str(ROOT/'repl_learning.py'),
        'lessons','--input',str(path),'--output',str(folder/'repl-lessons.json')],
        timeout=180*len(jobs))
    return read(folder/'repl-lessons.json')['rows']


def run_task(folder, task):
    path=folder/'repl-task.json'; output=folder/'repl-task-result.json'
    atomic_json(path,task)
    output.unlink(missing_ok=True)
    run_child([str(ROOT/'.cache/rlm-env/bin/python'),str(ROOT/'repl_learning.py'),
        'evaluate','--input',str(path),'--output',str(output)],timeout=LIMITS['seconds']+180)
    return read(output)


def evaluate(folder, tasks, adapter, phase, config):
    from coding_loop_stage import server, progress
    # Fixed family-separated validation tasks, evenly selected across available languages.
    from polyglot_runtime import LANGUAGES
    groups={l:[t for t in tasks if t['language']==l and t.get('kind')!='teacher_function'] for l in LANGUAGES}
    selected=[groups[l][i] for i in range(max(map(len,groups.values()),default=0))
              for l in LANGUAGES if i<len(groups[l])][:config.get('repl_tasks',10)]
    if not selected:
        raise ValueError('REPL comparison needs reserved program tasks')
    binding=dict(tasks_sha256=sha256(folder/'tasks.json'),ids=[t['id'] for t in selected],limits=LIMITS,
                 adapter_sha256=sha256(Path(adapter)/'adapter_model.safetensors') if adapter else None)
    path=folder/(phase+'-rows.json')
    saved=read(path,dict(binding=binding,rows=[]))
    if saved['binding']!=binding or [r['id'] for r in saved['rows']]!=binding['ids'][:len(saved['rows'])]:
        raise ValueError('Saved REPL evaluation tasks, weights or budgets changed')
    if len(saved['rows'])<len(selected):
        with server(adapter,folder):
            for task in selected[len(saved['rows']):]:
                if saved['rows'] and saved['rows'][-1].get('model_error','').startswith('TimeoutError:'):
                    # The single-request server may still finish a disconnected generation.
                    from evaluation import request
                    if request('http://127.0.0.1:8090','/health',timeout=180).get('status') != 'ok':
                        raise RuntimeError('REPL model server did not become idle')
                progress(folder,phase,len(saved['rows']),len(selected),task=task['id'],language=task['language'])
                result=run_task(folder,task)
                saved['rows'].append(dict(id=task['id'],language=task['language'],**result))
                atomic_json(path,saved)
    return dict(completed=True,**saved)


def evaluate_one(task):
    from recursive_agent import solve, grade
    from polyglot_runtime import check_program
    # Check the compiler independently so a broken sandbox cannot become a model failure.
    if not check_program('python','print(1)')['passed']:
        raise RuntimeError('REPL compiler runtime is unavailable')
    result=solve(task,'http://127.0.0.1:8090',mode='rlm',**LIMITS)
    error=result.get('error','')
    # The API timeout uses the remaining shared budget, so expiry is a scored attempt.
    if error.startswith('APITimeoutError:') and result['elapsed_s'] >= LIMITS['seconds']-1:
        error='TimeoutError: Model call reached the shared REPL deadline'
    if error.startswith('TimeoutExceededError:'):
        error='TimeoutError: '+error
    if error.startswith('ErrorThresholdExceededError:'):
        error='ValueError: '+error
    if error and not error.startswith(('ValueError:','JSONDecodeError:','TimeoutError:',
                                       'RuntimeError: Model call or output-token budget exhausted',
                                       'RuntimeError: Cell output limit exceeded')):
        raise RuntimeError('REPL evaluation failed: '+error)
    checked=grade(task,result['patch']) if result.get('patch') else dict(passed=False)
    return dict(passed=checked['passed'],calls=result['calls'],output_tokens=result['output_tokens'],
                elapsed_s=result['elapsed_s'],model_error=error,trace=result['trace'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('lessons','evaluate'))
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    if args.action=='lessons':build_lessons(read(args.input),args.output)
    else:atomic_json(args.output,evaluate_one(read(args.input)))


if __name__=='__main__':main()
