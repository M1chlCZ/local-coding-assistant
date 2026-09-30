"""Persistent REPL worker. Run only inside the restricted Docker container."""
import contextlib
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


state = {'__builtins__': __builtins__, 'answer': {'content': '', 'ready': False},
         'llm_query': query, 'rlm_query': lambda prompt, model=None: query(prompt, model, True)}
state['llm_query_batched'] = lambda prompts, model=None: [query(p, model) for p in prompts]
state['rlm_query_batched'] = lambda prompts, model=None: [query(p, model, True) for p in prompts]
state['SHOW_VARS'] = lambda: {k: type(v).__name__ for k, v in state.items() if not k.startswith('_')}
for line in wire_in:
    request = json.loads(line)
    if 'context' in request:
        state['context'] = request['context']
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
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(request['code'], state, state)
    except BaseException:
        try:
            traceback.print_exc(file=stderr)
        except RuntimeError:
            pass
    answer = state.get('answer', {})
    final = answer.get('content') if isinstance(answer, dict) and answer.get('ready') else None
    send({'type': 'result', 'stdout': stdout.getvalue(), 'stderr': stderr.getvalue(),
          'final_answer': json.dumps(final) if isinstance(final, dict) else final})
