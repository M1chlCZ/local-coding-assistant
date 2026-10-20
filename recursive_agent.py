"""Bounded local RLM runner and matched non-recursive REPL comparison."""
import argparse
import hashlib
import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from unittest.mock import patch

from evaluation import IMAGE, bounded_run, container_command

ROOT = Path(__file__).resolve().parent
UPSTREAM = 'd04208afbad29ca675ab13478c40ee8bebc84bfe'
TASKS = ROOT / 'research/rlm_tasks.json'
CALL_ORIGIN = ContextVar('rlm_call_origin', default=None)
MAX_ACTION_TOKENS = 1024
SYSTEM = '''You are an English coding assistant with a persistent isolated Python REPL.
The repository is in context['files'], a dictionary of relative paths to source text.
The task is in context['task']; editable paths are in context['editable'].
If context is a string, it is a recursive subtask. Inspect it and return the requested result (analysis or JSON). Do not assume repository files exist in the child workspace.
Inspect selected files by printing their contents. Execute Python only in ```repl``` blocks.
Files also exist under /workspace on the Python import path. You can import and test them.
Keep printed output short. Variables persist across cells. Do not use shell tools or network.
Return a JSON object mapping editable paths to their COMPLETE corrected source text.
Finish inside a repl block: answer['content'] = patch_dictionary; answer['ready'] = True.
Do not modify the context to pretend a repair succeeded. Hidden tests grade your final patch.
Do not print or return the whole repository. State no success without test evidence.
Inspect the relevant code, make the repair, and finish within four turns when possible. Avoid repetitive exploratory tests.
The variables context and answer already exist. Never initialize or replace context, or invent its contents.
Start with this bounded inspection. Read additional relevant files or source sections as needed:
```repl
if isinstance(context, dict):
    print(context['task'][:1500])
    print(context['editable'][:20])
    for name in context['editable'][:2]:
        print(name, context['files'][name][:2000])
else:
    print(context[:4000])
```
'''


def system_prompt(mode='rlm', instruction='', action_api=False):
    if action_api:
        from repl_actions import PROMPT, RECURSIVE
        return PROMPT + (RECURSIVE if mode=='rlm' else 'Subcalls are disabled. Repair and test in this REPL.\n') + instruction
    return SYSTEM + ('Use llm_query(prompt) or rlm_query(prompt) for focused analysis when useful. '
                     'Pass selected source text and a clear question, not the whole context.\n' if mode=='rlm'
                     else 'Subcalls are disabled. Analyze the files yourself in the REPL.\n') + instruction


def lesson_prompt(task):
    from rlm.core.rlm import QueryMetadata, build_rlm_system_prompt, build_user_prompt
    context = dict(files=task['files'], editable=task['editable'], task=task['prompt'],
                   **({'language': task['language']} if task.get('language') else {}))
    return build_rlm_system_prompt(system_prompt(), QueryMetadata(context),
        root_prompt=task['prompt'], orchestrator=False) + [build_user_prompt(max_iterations=8)]


def repl_feedback(stdout, stderr):
    return 'REPL output:\n' + ('\n\n'.join('\n'+part for part in (stdout, stderr) if part) or 'No output')


def format_iteration(iteration):
    from rlm.utils.parsing import format_iteration as original
    messages = original(iteration)
    if not iteration.code_blocks:
        messages.append(dict(role='user', content='No REPL command was executed. Use an executable '
            '```repl``` block, not a python block. context and answer already exist. '
            'For example:\n```repl\nprint(context["editable"][:20] if isinstance(context, dict) else context[:4000])\n```'))
    return messages


def valid_path(name):
    return (isinstance(name, str) and bool(name) and '\\' not in name and ':' not in name
            and not PurePosixPath(name).is_absolute()
            and all(p not in ('', '.', '..') and not p.startswith('.') for p in name.split('/')))


def parse_patch(answer, task):
    match = re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n```\s*', answer, re.S)
    value = json.loads(match.group(1) if match else answer)
    if not isinstance(value, dict) or not value or len(value) > len(task['editable']):
        raise ValueError('Answer must be a nonempty JSON patch object')
    for name, source in value.items():
        if (not valid_path(name) or name not in task['editable'] or name not in task['files']
                or not isinstance(source, str) or len(source) > 100000):
            raise ValueError('Patch contains an invalid path or source')
    return value


class Budget:
    def __init__(self, calls=16, output_tokens=8192, seconds=300):
        if not 1 <= calls <= 64 or not 1 <= output_tokens <= 65536 or not 0 < seconds <= 3600:
            raise ValueError('Invalid call, token or time limit')
        self.max_calls, self.max_output = calls, output_tokens
        self.deadline = time.monotonic() + seconds
        self.calls, self.output_tokens, self.input_tokens = 0, 0, 0
        self.trace = []
        self.queries, self.executions = [], []
        self.infrastructure_errors = []
        self.lock = threading.Lock()

    def remaining(self):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError('Experiment time limit reached')
        return left

    def reserve(self, requested):
        self.remaining()
        if self.calls >= self.max_calls or self.output_tokens >= self.max_output:
            raise RuntimeError('Model call or output-token budget exhausted')
        self.calls += 1
        return min(requested, self.max_output - self.output_tokens)


class Sandbox:
    """Upstream-compatible environment with no network, mounts or host-code execution."""
    def __init__(self, context_payload, lm_handler_address=None, depth=1, subcall_fn=None,
                 allow_queries=True, cell_seconds=20, budget=None, action_api=False, **kwargs):
        self.address, self.depth, self.subcall = lm_handler_address, depth, subcall_fn
        self.allow_queries, self.cell_seconds, self.budget = allow_queries, cell_seconds, budget
        self.action_api = action_api
        self.editable = context_payload.get('editable', []) if isinstance(context_payload, dict) else []
        self.workspace_hashes = {}
        self.container = None
        self.process = None
        self.inbox = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.reader = self.writer = None
        name = 'local-rlm-' + uuid.uuid4().hex
        self.container = name  # Retain ownership even if Docker creation times out.
        if isinstance(context_payload, dict) and context_payload.get('language'):
            from polyglot_runtime import container_command as native_command
            command = native_command()
        else:
            command = container_command(IMAGE)
        command[2:2] = ['--name', name, '--label', 'local-coding-assistant.rlm=true']
        if os.environ.get('LCA_REPL_OWNER'):
            command[2:2] = ['--label','local-coding-assistant.owner='+os.environ['LCA_REPL_OWNER']]
        try:
            self.container = subprocess.check_output(command + ['sleep', '3600'], text=True, timeout=30).strip()
            subprocess.run(['docker', 'exec', '-i', self.container, 'python', '-I', '-c',
                            'import pathlib,sys;pathlib.Path("/workspace/worker.py").write_text(sys.stdin.read())'],
                           input=(ROOT / 'rlm_worker.py').read_text(), text=True,
                           check=True, capture_output=True, timeout=10)
            if action_api:
                subprocess.run(['docker', 'exec', '-i', self.container, 'python', '-I', '-c',
                    'import pathlib,sys;pathlib.Path("/workspace/repl_actions.py").write_text(sys.stdin.read())'],
                    input=(ROOT/'repl_actions.py').read_text(), text=True, check=True,
                    capture_output=True, timeout=10)
            self.process = subprocess.Popen(['docker', 'exec', '-i', '-w', '/workspace', self.container, 'python', '-I', '-u',
                                             '/workspace/worker.py'], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.reader = threading.Thread(target=self._read, daemon=True, name='local-rlm-reader')
            self.reader.start()
            self._send({'context': context_payload, **({'action_api': True} if action_api else {})})
            self._receive()
        except BaseException:
            self.cleanup()
            raise

    def _read(self):
        stream = self.process.stdout
        while not self.stopped.is_set():
            line = stream.readline(131073)
            while not self.stopped.is_set():
                try:
                    self.inbox.put(line, timeout=.1)
                    break
                except queue.Full:
                    pass
            if not line or len(line) > 131072:
                break

    def _send(self, value):
        if self.stopped.is_set() or self.writer and self.writer.is_alive():
            raise TimeoutError('REPL pipe is stopped or blocked')
        seconds = min(self.cell_seconds, self.budget.remaining()) if self.budget else self.cell_seconds
        data = (json.dumps(value) + '\n').encode()
        stream = self.process.stdin
        errors = []

        def send():
            try:
                stream.write(data)
                stream.flush()
            except OSError as error:
                errors.append(error)

        self.writer = threading.Thread(target=send, daemon=True, name='local-rlm-writer')
        self.writer.start()
        self.writer.join(timeout=seconds)
        if self.writer.is_alive():
            raise TimeoutError('REPL pipe write time limit reached')
        if errors:
            raise errors[0]

    def _receive(self):
        seconds = min(self.cell_seconds, self.budget.remaining()) if self.budget else self.cell_seconds
        try:
            line = self.inbox.get(timeout=seconds)
        except queue.Empty:
            raise TimeoutError('REPL cell time limit reached') from None
        if not line or len(line) > 131072:
            raise RuntimeError('REPL exited or exceeded the protocol output limit')
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError('Invalid REPL message')
        return value

    def execute_code(self, code):
        from rlm.core.comms_utils import LMRequest, send_lm_request
        from rlm.core.types import REPLResult
        if not isinstance(code, str) or len(code) > 100000:
            raise ValueError('REPL cell is too large')
        started = time.monotonic()
        self._send({'code': code})
        calls = []
        while True:
            frame = self._receive()
            if frame.get('type') == 'result':
                if set(frame) != {'type', 'stdout', 'stderr', 'final_answer'} | ({'checks'} if self.action_api else set()):
                    raise ValueError('Invalid REPL result')
                if not all(isinstance(frame[k], str) and len(frame[k]) <= 8192 for k in ('stdout', 'stderr')):
                    raise ValueError('Invalid or oversized REPL output')
                if (frame['final_answer'] is not None and
                        (not isinstance(frame['final_answer'], str) or len(frame['final_answer']) > 100000)):
                    raise ValueError('Invalid model final answer')
                if 'Cell output limit exceeded' in frame['stderr']:
                    raise RuntimeError('Cell output limit exceeded')
                if self.budget:
                    self.budget.executions.append(dict(depth=self.depth, code=code,
                        stderr=frame['stderr'], final_answer=frame['final_answer'],
                        **({'checks': frame['checks']} if self.action_api else {})))
                if frame['final_answer'] is not None and self.editable:
                    source = ('import pathlib,json,hashlib; print(json.dumps({name: '
                              'hashlib.sha256(pathlib.Path("/workspace",name).read_bytes()).hexdigest() '
                              'if pathlib.Path("/workspace",name).is_file() else None '
                              'for name in '+repr([n for n in self.editable if valid_path(n)])+'}))')
                    self.workspace_hashes = json.loads(subprocess.check_output(
                        ['docker','exec',self.container,'python','-I','-c',source],text=True,timeout=10))
                return REPLResult(stdout=frame['stdout'], stderr=frame['stderr'], locals={},
                                  execution_time=time.monotonic()-started, rlm_calls=calls,
                                  final_answer=frame['final_answer'])
            if (frame.get('type') != 'query' or set(frame) != {'type', 'prompt', 'recursive'}
                    or not isinstance(frame['prompt'], str) or len(frame['prompt']) > 16000
                    or not isinstance(frame['recursive'], bool)):
                raise ValueError('Invalid subcall message')
            try:
                if not self.allow_queries:
                    raise RuntimeError('Subcalls are disabled in the baseline')
                recursive = bool(frame['recursive'] and self.subcall)
                first_execution = len(self.budget.executions) if self.budget else 0
                if recursive:
                    result = self.subcall(frame['prompt'])
                else:
                    reply = send_lm_request(self.address, LMRequest(prompt=frame['prompt'], depth=self.depth),
                                            timeout=self.budget.remaining() if self.budget else self.cell_seconds)
                    if not reply.success or reply.chat_completion is None:
                        raise RuntimeError(reply.error or 'Subcall returned no completion')
                    result = reply.chat_completion
                calls.append(result)
                if self.budget:
                    self.budget.queries.append(dict(recursive=recursive, response=result.response,
                        execution_start=first_execution, execution_end=len(self.budget.executions)))
                self._send({'response': result.response})
            except Exception as error:
                self._send({'error': str(error)[:1000]})

    def cleanup(self):
        self.stopped.set()
        if self.process:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=5)
        try:
            if self.container:
                subprocess.run(['docker', 'rm', '-f', self.container], capture_output=True, timeout=15)
                self.container = None
        finally:
            for thread in (self.writer, self.reader):
                if thread:
                    thread.join(timeout=2)
            if self.process:
                for stream in (self.process.stdin, self.process.stdout):
                    try:
                        stream.close()
                    except OSError:
                        pass  # A killed worker may leave buffered stdin with no reader.
                self.process = None


def solve(task, base, mode='rlm', depth=1, calls=16, output_tokens=8192, seconds=300, instruction='', action_api=False):
    from rlm import RLM
    from rlm.clients.openai import OpenAIClient
    from rlm.core import rlm as core
    from rlm.core.lm_handler import LMRequestHandler
    from rlm.logger import RLMLogger
    url = urlparse(base)
    if url.scheme != 'http' or url.hostname not in ('localhost', '127.0.0.1') or url.username or url.password:
        raise ValueError('Use the local loopback model API or an SSH relay')
    if mode not in ('baseline', 'rlm') or depth not in (1, 2):
        raise ValueError('Use baseline or rlm, with recursion depth 1 or 2')
    if not isinstance(instruction, str) or len(instruction)>2000:
        raise ValueError('Experiment instruction exceeds 2000 characters')
    budget = Budget(calls, output_tokens, seconds)

    class LocalClient(OpenAIClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.origin = CALL_ORIGIN.get() or 'root'

        def completion(self, prompt, model=None):
            with budget.lock:
                maximum = budget.reserve(MAX_ACTION_TOKENS)
                self.sampling_args['max_completion_tokens'] = maximum
                self.client.timeout = budget.remaining()
                try:
                    response = super().completion(prompt, model)
                except Exception as error:
                    if not (type(error).__name__=='APITimeoutError' and budget.deadline-time.monotonic()<=1):
                        budget.infrastructure_errors.append(f'{type(error).__name__}: {error}')
                    raise
                used = self.get_last_usage()
                budget.input_tokens += used.total_input_tokens
                budget.output_tokens += used.total_output_tokens
                budget.trace.append({'messages': json.loads(json.dumps(prompt)) if isinstance(prompt, list) else
                                     [{'role': 'user', 'content': prompt}], 'response': response,
                                     'input_tokens': used.total_input_tokens, 'output_tokens': used.total_output_tokens,
                                     'kind': CALL_ORIGIN.get() or self.origin})
                return response

    environments = []
    original_context = RLM._spawn_completion_context
    original_subcall = RLM._subcall
    original_single = LMRequestHandler._handle_single

    @contextmanager
    def completion_context(self, prompt):
        token = CALL_ORIGIN.set('root' if self.depth == 0 else 'subcall')
        try:
            with original_context(self, prompt) as value:
                yield value
        finally:
            CALL_ORIGIN.reset(token)

    def subcall(self, *args, **kwargs):
        token = CALL_ORIGIN.set('subcall')
        try:
            return original_subcall(self, *args, **kwargs)
        finally:
            CALL_ORIGIN.reset(token)

    def single(self, *args, **kwargs):
        token = CALL_ORIGIN.set('subcall')
        try:
            return original_single(self, *args, **kwargs)
        finally:
            CALL_ORIGIN.reset(token)

    class ObservedSandbox(Sandbox):
        def execute_code(self, code):
            try:
                return super().execute_code(code)
            except Exception as error:
                if (not isinstance(error, TimeoutError) and str(error)!='Cell output limit exceeded'
                        and not (isinstance(error, ValueError) and str(error)=='Invalid model final answer')):
                    budget.infrastructure_errors.append(f'{type(error).__name__}: {error}')
                raise

    def environment(name, kwargs):
        try:
            env = ObservedSandbox(**kwargs, allow_queries=mode=='rlm', budget=budget, action_api=action_api)
        except Exception as error:
            budget.infrastructure_errors.append(f'{type(error).__name__}: {error}')
            raise
        environments.append(env)
        return env

    system = system_prompt(mode, instruction, action_api)
    started = time.monotonic()
    row = {'id': task['id'], 'repository': task['repository'], 'split': task['split'],
           'mode': mode, 'depth': depth, 'passed': False}
    if action_api:
        row['action_api'] = True
    try:
        # ponytail: scope sandbox, trace-origin and fallback patches to the pinned upstream; use public hooks if provided later.
        with patch.object(core, 'get_environment', environment), \
             patch.object(core, 'get_client', lambda backend, kwargs: LocalClient(**kwargs)), \
             patch.object(core, 'format_iteration', format_iteration), \
             patch.object(RLM, '_spawn_completion_context', completion_context), \
             patch.object(RLM, '_subcall', subcall), \
             patch.object(LMRequestHandler, '_handle_single', single), \
             patch.object(RLM, '_default_answer', lambda self, history, handler: handler.completion(history + [
                 {'role':'user','content':'Return the final requested answer now. For a repository repair, return ONLY the JSON patch object. For a text subtask, return the requested analysis.'}])):
            runner = RLM(backend='openai', backend_kwargs={'base_url': base.rstrip('/')+'/v1',
                         'model_name': 'local-coding-assistant', 'api_key': 'local', 'max_retries': 0,
                         'timeout': seconds}, environment='docker', max_depth=depth, max_iterations=8,
                         max_timeout=seconds, max_errors=3, max_concurrent_subcalls=1,
                         custom_system_prompt=system, orchestrator=False, logger=RLMLogger(),
                         sampling_args={'temperature': 0, 'max_completion_tokens': MAX_ACTION_TOKENS,
                         'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}})
            result = runner.completion({'files': task['files'], 'editable': task['editable'],
                                        'task': task['prompt'], **({'language':task['language']} if task.get('language') else {})}, root_prompt=task['prompt'])
            row['answer'] = result.response
            row['patch'] = parse_patch(result.response, task)
            row['workspace_hashes'] = {name:environments[0].workspace_hashes.get(name) for name in row['patch']}
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
    finally:
        for env in environments:
            env.cleanup()
    if budget.infrastructure_errors:
        raise RuntimeError('Infrastructure failure: '+'; '.join(budget.infrastructure_errors))
    row.update(elapsed_s=round(time.monotonic()-started,3), calls=budget.calls,
               input_tokens=budget.input_tokens, output_tokens=budget.output_tokens,
               trace=budget.trace, queries=budget.queries, executions=budget.executions,
               infrastructure_errors=budget.infrastructure_errors)
    return row


def grade(task, changes):
    """Functional checks, not a hostile-code verifier: imported Python shares the grader process."""
    changes = parse_patch(json.dumps(changes), task)
    if task.get('language'):
        from polyglot_runtime import grade as native_grade
        return native_grade(task, changes)
    files = {**task['files'], **changes}
    marker = 'RLM_CHECKS_PASSED_' + uuid.uuid4().hex
    program = ('import json,pathlib,sys,traceback\nfiles=json.loads('+repr(json.dumps(files))+')\n'
               'for name,content in files.items():\n p=pathlib.Path("/workspace")/name; '
               'p.parent.mkdir(parents=True,exist_ok=True); p.write_text(content)\n'
               'sys.path.insert(0,"/workspace")\n'
               'try:\n exec('+repr(task['checks'])+',{})\n'
               'except BaseException:\n traceback.print_exc(); raise SystemExit(1)\n'
               'print('+repr(marker)+')\n')
    name = 'local-rlm-grade-' + uuid.uuid4().hex
    command = container_command(IMAGE, True)
    command[2:2] = ['--name', name]
    if os.environ.get('LCA_REPL_OWNER'):
        command[2:2] = ['--label','local-coding-assistant.owner='+os.environ['LCA_REPL_OWNER']]
    try:
        result = bounded_run(command+['python','-I','-S','-'], program, timeout=20)
        if result['exit_code'] in (125, 126, 127):
            raise RuntimeError('Grader container infrastructure failed: '+result['output'])
        return {'passed':result['exit_code']==0 and not result['output_limited']
                and result['output'].strip().endswith(marker),
                'feedback':result['output'].replace(marker, 'RLM_CHECKS_PASSED')}
    finally:
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=15)


def snapshot(root):
    files = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        if not valid_path(relative) or path.is_symlink() or path.suffix not in ('.py','.md','.go','.ts','.tsx','.rs','.dart'):
            continue
        if any((root.joinpath(*Path(relative).parts[:n])).is_symlink() for n in range(1,len(Path(relative).parts))):
            continue
        if path.is_file():
            if path.stat().st_size > 100000:
                raise ValueError('Repository file exceeds 100000 bytes: '+relative)
            files[relative] = path.read_text(encoding='utf-8')
            if len(files)>256 or sum(len(x.encode()) for x in files.values())>2*1024*1024:
                raise ValueError('Repository snapshot exceeds 256 files or 2 MiB')
    if not files:
        raise ValueError('No readable source or Markdown files')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8080')
    parser.add_argument('--mode', choices=['baseline','rlm'], default='rlm')
    parser.add_argument('--depth', type=int, choices=[1,2], default=1)
    parser.add_argument('--calls',type=int,default=16)
    parser.add_argument('--output-tokens',type=int,default=8192)
    parser.add_argument('--seconds',type=int,default=300)
    parser.add_argument('--split',choices=['train','dev','holdout'],default='dev')
    parser.add_argument('--tasks',type=Path,default=TASKS,help='Local repair task registry')
    parser.add_argument('--task', help='Run one fixture ID')
    parser.add_argument('--repo',type=Path,help='Read a repository snapshot; output a proposed patch without applying it')
    parser.add_argument('--prompt',help='Repair request for --repo')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new report path')
    if args.repo:
        if not args.prompt:
            parser.error('--repo requires --prompt')
        files=snapshot(args.repo.resolve())
        tasks=[{'id':'custom','repository':args.repo.name,'split':'custom','prompt':args.prompt,
                'files':files,'editable':[f for f in files if f.endswith('.py')]}]
        digest=None
    else:
        tasks=json.loads(args.tasks.read_text())
        tasks=[t for t in tasks if t['split']==args.split and (not args.task or t['id']==args.task)]
        digest=hashlib.sha256(args.tasks.read_bytes()).hexdigest()
        if not tasks:
            parser.error('No tasks match this split and ID')
    report={'schema_version':1,'split':'custom' if args.repo else args.split,'tasks_sha256':digest,
            'upstream_revision':UPSTREAM,'limits':{'calls':args.calls,'output_tokens':args.output_tokens,
            'seconds':args.seconds},'scope':'Local repair tasks from the bound registry, one graded patch attempt. No general agent reliability claim.',
            'tasks':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for task in tasks:
        row=solve(task,args.base,args.mode,args.depth,args.calls,args.output_tokens,args.seconds)
        if row.get('patch') and not args.repo:
            try:
                row.update(grade(task,row['patch']))
            except Exception as error:
                row.update(passed=False, feedback=f'{type(error).__name__}: {error}')
        report['tasks'].append(row)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(f"{task['id']}: {'PASS' if row['passed'] else 'unverified' if args.repo else 'FAIL'}; {row['calls']} calls, {row['elapsed_s']}s",flush=True)


if __name__=='__main__':
    main()
