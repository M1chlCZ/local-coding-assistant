"""Research-only typed actions over the existing isolated REPL, with no fallback answer."""
import json
import time
from types import SimpleNamespace
from urllib.parse import urlparse

from evaluation import request
from recursive_agent import Budget, CALL_ORIGIN, ContextWindowExceeded, Sandbox, parse_patch, valid_path
from structured_output import check_response

TOOLS = ('show', 'read', 'write', 'check', 'delegate', 'apply', 'finish')
PROMPT = '''Repair the repository through one tool action per response.
Return ONLY JSON with exactly three string fields: tool, path, text. Use empty strings for unused fields.
show: empty path/text; displays requirements and source.
read: path is a repository path; text empty.
write: path is an editable path; text is its COMPLETE Python source. Syntax is validated before writing.
check: path empty; text is Python code with actual assertions derived from the requirements.
Checks run in a fresh process with repository imports. No tests, pytest, or test functions are supplied.
delegate: path empty; text is a focused repair goal. A child inspects, checks, repairs, checks again,
and finishes. Its completed patch is saved as the pending proposal; you do not parse JSON or manage variables.
apply: empty path/text; applies the pending child proposal.
finish: empty path/text; returns changed source from disk only after a successful check on those files.
Inspect the source. Reproduce a defect with assertions before editing, repair, then check again.
Correct errors using their feedback. Do not repeat unchanged failures. Do not invent test functions.
Only write/check text contains Python; never return executable dispatch code or a final patch yourself.
'''


def schema(child):
    tools = [t for t in TOOLS if not child or t not in ('delegate', 'apply')]
    return dict(type='object', properties={
        'tool': dict(type='string', enum=tools), 'path': dict(type='string'),
        'text': dict(type='string')}, required=['tool', 'path', 'text'], additionalProperties=False)


def action_code(response, child):
    check_response(response, schema(child))
    if not response['structured_output']['valid']:
        raise ValueError('Invalid or incomplete action: '+response['structured_output']['error'])
    action = json.loads(response['choices'][0]['message']['content'])
    tool, path, text = (action[k] for k in ('tool', 'path', 'text'))
    if len(text) > 100000 or (path and not valid_path(path)):
        raise ValueError('Invalid path or oversized action')
    if tool in ('show', 'apply', 'finish'):
        if path or text:
            raise ValueError('This action takes no path or text')
        return 'apply(proposal)' if tool == 'apply' else tool+'()'
    if tool == 'read':
        if not path or text:
            raise ValueError('read requires only a path')
        return 'print(read('+repr(path)+'))'
    if tool == 'write':
        if not path:
            raise ValueError('write requires a path')
        return 'write('+repr(path)+', '+repr(text)+')'
    if path or not text or len(text) > (2000 if tool == 'delegate' else 12000):
        raise ValueError('check/delegate require bounded nonempty text and no path')
    return ('proposal = delegate(' if tool == 'delegate' else 'check(')+repr(text)+')'


def solve(task, base, *, recursive=False, constrained=False, calls=12, output_tokens=8192, seconds=360):
    url = urlparse(base)
    if url.scheme != 'http' or url.hostname not in ('localhost', '127.0.0.1') or url.username or url.password:
        raise ValueError('Use a local loopback model server')
    budget = Budget(calls, output_tokens, seconds)
    started = time.monotonic()
    row = dict(id=task['id'], repository=task['repository'], split=task['split'],
               passed=False, action_api=True, protocol_errors=[])

    def complete(context, child=False):
        token = CALL_ORIGIN.set('subcall' if child else 'root')
        environment = None
        try:
            try:
                environment = Sandbox(context_payload=context, depth=2 if child else 1,
                    subcall_fn=lambda prompt: complete(prompt, True), allow_queries=recursive and not child,
                    budget=budget, action_api=True)
            except Exception as error:
                budget.infrastructure_errors.append(f'{type(error).__name__}: {error}')
                raise
            no_delegation = child or not recursive
            instructions = (('You are the child; do not delegate. ' if child else
                'Solve independently; delegation is disabled. ') if no_delegation else
                'You are the parent. After inspection delegate one focused repair, apply its proposal, '
                'then independently check the repaired source and finish. ')
            payload = json.loads(context) if isinstance(context, str) else context
            messages = [dict(role='system', content=PROMPT+instructions+
                            '\nAction schema: '+json.dumps(schema(no_delegation))),
                        dict(role='user', content=json.dumps(payload, ensure_ascii=False))]
            for _ in range(8):
                maximum = budget.reserve(1024)
                try:
                    messages = budget.prepare_prompt(messages, base, maximum)
                    data = dict(model='local-coding-assistant', messages=messages,
                                temperature=0, max_completion_tokens=maximum,
                                chat_template_kwargs=dict(enable_thinking=False))
                    if constrained:
                        data['response_format'] = dict(type='json_schema', json_schema=dict(
                            name='action', strict=True, schema=schema(no_delegation)))
                    response = request(base, '/v1/chat/completions', data, timeout=budget.remaining())
                    usage = response['usage']
                    content = response['choices'][0]['message']['content']
                    budget.input_tokens += usage['prompt_tokens']
                    budget.output_tokens += usage['completion_tokens']
                except Exception as error:
                    if not isinstance(error, ContextWindowExceeded) and not (
                            isinstance(error, TimeoutError) and time.monotonic() >= budget.deadline-1):
                        budget.infrastructure_errors.append(f'{type(error).__name__}: {error}')
                    raise
                budget.trace.append(dict(messages=json.loads(json.dumps(messages)), response=content,
                    raw_response=response, input_tokens=usage['prompt_tokens'],
                    output_tokens=usage['completion_tokens'], kind=CALL_ORIGIN.get()))
                messages.append(dict(role='assistant', content=content))
                try:
                    code = action_code(response, no_delegation)
                except ValueError as error:
                    row['protocol_errors'].append(dict(kind=CALL_ORIGIN.get(), error=str(error)))
                    messages.append(dict(role='user', content='Action rejected: '+str(error)))
                    continue
                try:
                    result = environment.execute_code(code)
                except Exception as error:
                    if not isinstance(error, TimeoutError) and str(error) != 'Cell output limit exceeded':
                        budget.infrastructure_errors.append(f'{type(error).__name__}: {error}')
                    raise
                messages.append(dict(role='user', content='Tool result:\n'+result.stdout+'\n'+result.stderr))
                if result.final_answer is not None and not result.stderr:
                    # A completion is an observed worker event, never a model's claim or fallback text.
                    if not child:
                        row['workspace_hashes'] = dict(environment.workspace_hashes)
                    return SimpleNamespace(response=result.final_answer)
            raise RuntimeError('Tool action limit reached without a checked finish')
        finally:
            if environment:
                environment.cleanup()
            CALL_ORIGIN.reset(token)

    try:
        result = complete(dict(task=task['prompt'], files=task['files'], editable=task['editable']))
        row.update(answer=result.response, patch=parse_patch(result.response, task))
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
    if budget.infrastructure_errors:
        raise RuntimeError('Infrastructure failure: '+'; '.join(budget.infrastructure_errors))
    row.update(elapsed_s=round(time.monotonic()-started, 3), calls=budget.calls,
               input_tokens=budget.input_tokens, output_tokens=budget.output_tokens,
               trace=budget.trace, queries=budget.queries, executions=budget.executions,
               infrastructure_errors=budget.infrastructure_errors, context_windows=budget.context_windows,
               context_exhausted=budget.context_exhausted)
    return row
