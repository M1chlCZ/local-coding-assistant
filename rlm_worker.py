"""Persistent REPL worker. Run only inside the restricted Docker container."""
import ast
import contextlib
import hashlib
import io
import json
import pathlib
import sys
import traceback

wire_in, wire_out = sys.stdin, sys.stdout


def send(value):
    wire_out.write(json.dumps(value) + '\n')
    wire_out.flush()


class Output(io.StringIO):
    def write(self, text):
        if self.tell() + len(text) > 8192:
            raise RuntimeError('Cell output limit exceeded')
        return super().write(text)


def query(prompt, model=None, recursive=False):
    if model is not None:
        raise ValueError('This runner uses one local model')
    if not isinstance(prompt, str) or len(prompt) > 16000:
        raise ValueError('Subcall needs a string of at most 16000 characters')
    send({'type': 'query', 'prompt': prompt, 'recursive': recursive})
    reply = json.loads(wire_in.readline(131073))
    if reply.get('error'):
        raise RuntimeError(reply['error'])
    return reply['response']


def workspace_hashes(context):
    """Hash editable files so fenced cells can leave the same evidence as actions."""
    if not isinstance(context, dict):
        return {}
    editable = context.get('editable', [])
    if not isinstance(editable, list):
        return {}
    result = {}
    for name in editable:
        if not isinstance(name, str) or not name or '\\' in name or ':' in name:
            continue
        path = pathlib.PurePosixPath(name)
        if (path.is_absolute() or any(not part or part.startswith('.')
                                      for part in path.parts)):
            continue
        target = pathlib.Path('/workspace') / path
        if target.is_file():
            result[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return result


class RecordAssertions(ast.NodeTransformer):
    def visit_Assert(self, node):
        # Keep assert's lazy message evaluation and record only statements reached at runtime.
        def record(passed):
            return ast.copy_location(ast.Expr(ast.Call(
                ast.Name('_repl_record_assertion', ast.Load()),
                [ast.Constant(passed)], [])), node)
        return [record(False), node, record(True)]


state = {'__builtins__': __builtins__, 'answer': {'content': '', 'ready': False},
         'llm_query': query, 'rlm_query': lambda prompt, model=None: query(prompt, model, True)}
state['llm_query_batched'] = lambda prompts, model=None: [query(p, model) for p in prompts]
state['rlm_query_batched'] = lambda prompts, model=None: [query(p, model, True) for p in prompts]
state['SHOW_VARS'] = lambda: {k: type(v).__name__ for k, v in state.items() if not k.startswith('_')}
actions = None
action_api = False
for line in wire_in:
    request = json.loads(line)
    if 'context' in request:
        state['context'] = request['context']
        action_api = bool(request.get('action_api'))
        repository = isinstance(request['context'], dict) and isinstance(
            request['context'].get('files'), dict)
        if action_api or repository:
            sys.path.insert(0, '/workspace')
            from repl_actions import Workspace
            actions = Workspace(request['context'], state['answer'],
                                query=lambda prompt: query(prompt, recursive=True))
            if action_api:
                state.update(actions.bindings())
            else:
                # Fenced cells may use the same check helper without the full action prompt.
                state['check'] = actions.check
        if isinstance(request['context'], dict):
            for name, content in request['context'].get('files', {}).items():
                path = pathlib.PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or '\\' in name:
                    raise ValueError('Invalid context path')
                target = pathlib.Path('/workspace') / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            sys.path.insert(0, '/workspace')
        send({'type': 'result', 'stdout': '', 'stderr': '', 'final_answer': None})
        continue
    stdout, stderr = Output(), Output()
    context = state.get('context')
    recorded = dict(attempted=0, passed=0, pending=[], checked=None, stable=True)

    def record_assertion(passed):
        hashes = workspace_hashes(context)
        if not passed:
            recorded['attempted'] += 1
            recorded['pending'].append(hashes)
        else:
            recorded['passed'] += 1
            recorded['stable'] &= recorded['pending'].pop() == hashes
            recorded['checked'] = hashes

    state['_repl_record_assertion'] = record_assertion
    first_check = len(actions.checks) if actions else 0
    failed = False
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = request['code']
            if not action_api:
                code = compile(ast.fix_missing_locations(
                    RecordAssertions().visit(ast.parse(code))), '<repl>', 'exec')
            exec(code, state, state)
    except BaseException:
        failed = True
        try:
            traceback.print_exc(file=stderr)
        except RuntimeError:
            pass
    after = workspace_hashes(context)
    checks = list(actions.checks[first_check:]) if actions else []
    if not action_api:
        assertions = recorded['attempted']
        if assertions:
            output = (stdout.getvalue()+stderr.getvalue())[-4096:]
            checks.append(dict(code=request['code'][:12000],
                returncode=int(failed or bool(stderr.getvalue())
                    or recorded['passed'] != assertions or not recorded['stable']),
                output=output, assertions=assertions,
                before=recorded['checked'], after=after))
    answer = state.get('answer', {})
    final = answer.get('content') if isinstance(answer, dict) and answer.get('ready') else None
    send({'type': 'result', 'stdout': stdout.getvalue(), 'stderr': stderr.getvalue(),
          'final_answer': json.dumps(final) if isinstance(final, dict) else final,
          'checks': checks})
