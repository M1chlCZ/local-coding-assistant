"""Same retained weights, prompts and budgets; vary only JSON-schema constraints."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import time

from qwen35_train import MODEL, atomic_json, load_model, read_json, sha256, validate_revision
from qwen35_server import generate, stop_tokens
from recursive_agent import grade
from structured_output import StructuredOutput, check_response

ROOT = Path(__file__).resolve().parents[1]
TASKS = Path(__file__).with_name('structured_tasks.json')


def schema_for(task):
    return {'type': 'object', 'properties': {p: {'type': 'string'} for p in task['editable']},
            'required': task['editable'], 'additionalProperties': False}


def evaluate(task, response, elapsed, grader=grade):
    check_response(response, schema_for(task))
    row = dict(id=task['id'], response=response, elapsed_s=round(elapsed, 3),
               format_valid=response['structured_output']['valid'],
               syntax_valid=False, functional_passed=False)
    if not row['format_valid']:
        return row
    patch = json.loads(response['choices'][0]['message']['content'])
    try:
        for name, code in patch.items():
            compile(code, name, 'exec')
    except (SyntaxError, ValueError):
        return row
    row['syntax_valid'] = True
    row['grading'] = grader(task, patch)  # Infrastructure failures propagate.
    row['functional_passed'] = row['grading']['passed']
    return row


def summarize(rows, ids):
    expected = {(name, mode) for name in ids for mode in ('plain', 'constrained')}
    indexed = {(row['id'], row['mode']): row for row in rows}
    if len(rows) != len(expected) or len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError('A complete unique matched pair is required for every task')
    result = {'completed': True, 'tasks_per_mode': len(ids), 'modes': {},
              'learning_gain_established': False, 'full_loop_authorized': False,
              'scope': 'Four public direct-patch format canaries; no REPL, recursive or held-out learning claim.'}
    for mode in ('plain', 'constrained'):
        selected = [indexed[name, mode] for name in ids]
        result['modes'][mode] = {key: sum(row[key] for row in selected)
                                for key in ('format_valid', 'syntax_valid', 'functional_passed', 'elapsed_s')}
    for suffix, before, after in [('gains', False, True), ('losses', True, False)]:
        result['functional_'+suffix] = [name for name in ids
            if indexed[name, 'plain']['functional_passed'] == before
            and indexed[name, 'constrained']['functional_passed'] == after]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-lock', type=Path, required=True,
                        help='Shared lock used by the existing training installation')
    parser.add_argument('--after-session', type=Path, required=True,
                        help='Existing finite experiment must be completed and its worker stopped')
    args = parser.parse_args()
    validate_revision(MODEL, args.revision)
    from focused_experiment import locked
    session = args.after_session
    if read_json(session/'status.json')['status'] != 'completed' or locked(session/'worker.lock'):
        raise RuntimeError('The existing experiment still owns its GPU schedule')
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; partial evidence is preserved')
    import fcntl
    with ExitStack() as stack:
        for path in sorted({args.gpu_lock.resolve(), (session.parent/'gpu.lock').resolve()}):
            stream = stack.enter_context(path.open('a'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for port in (8080, 8090):
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', port)) == 0:
                    raise RuntimeError('An existing model server is still running')
        tasks = read_json(TASKS)
        for task in tasks:
            if grade(task, {p:task['files'][p] for p in task['editable']})['passed']:
                raise ValueError('A canary incorrectly accepts the unchanged source')
        args.output.mkdir(parents=True)
        sources = ['research/structured_compare.py', 'research/structured_tasks.json', 'structured_output.py',
                   'qwen35_server.py', 'qwen35_train.py', 'recursive_agent.py', 'evaluation.py',
                   'research/adapter_eval_server.py', 'requirements-structured.txt']
        binding = dict(model=MODEL, revision=args.revision, adapter=None, precision='bf16',
            max_tokens=1024, context=8192, temperature=0, generation_batch_size=1,
            sources={name:sha256(ROOT/name) for name in sources},
            runtime={p:importlib.metadata.version(p) for p in ('torch','transformers','unsloth','xgrammar','jsonschema')})
        atomic_json(args.output/'binding.json', binding)
        model, tokenizer, fast = load_model(MODEL, args.revision, 8192)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        fast.for_inference(model)
        model.eval()
        model.config.use_cache = True
        engine = StructuredOutput(tokenizer, model.get_output_embeddings().weight.shape[0],
                                  stop_tokens(model, tokenizer))
        rows = []
        for index, task in enumerate(tasks):
            schema = schema_for(task)
            messages = [
                {'role':'system', 'content':'Repair the supplied repository. Return ONLY a JSON object '
                 'with the complete corrected source for every editable file, matching this schema: '+json.dumps(schema)},
                {'role':'user', 'content':task['prompt']+'\nSource files:\n'+json.dumps(task['files'], ensure_ascii=False)},
            ]
            prompt_sha = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
            for mode in (('plain','constrained') if index % 2 == 0 else ('constrained','plain')):
                data = {'conversations':[messages], 'max_tokens':1024, 'temperature':0}
                if mode == 'constrained':
                    data['response_format'] = {'type':'json_schema', 'json_schema':{
                        'name':'patch', 'strict':True, 'schema':schema}}
                started = time.monotonic()
                response = generate(model, tokenizer, data, False, False, 1024, 8192, 1, engine)[0]
                row = evaluate(task, response, time.monotonic()-started)
                rows.append({**row, 'mode':mode, 'prompt_sha256':prompt_sha})
                atomic_json(args.output/'results.json', {'completed':False, 'rows':rows})
                print(task['id'], mode, 'format', row['format_valid'],
                      'functional', row['functional_passed'], flush=True)
        if binding['sources'] != {name:sha256(ROOT/name) for name in sources}:
            raise RuntimeError('Comparison source changed during execution')
        summary = summarize(rows, [task['id'] for task in tasks])
        atomic_json(args.output/'results.json', {'completed':True, 'rows':rows})
        atomic_json(args.output/'summary.json', {**summary, 'binding':binding})


if __name__ == '__main__':
    main()
