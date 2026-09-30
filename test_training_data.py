"""Run with python test_training_data.py. No GPU or extra packages needed."""
import copy
import json
import tempfile
from pathlib import Path

import training_data as data
import train_adapter


def rejected(call):
    try:
        call()
    except (ValueError, FileExistsError):
        return
    raise AssertionError('Invalid input was accepted')


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    registry = root / 'tasks.json'
    tasks = [{'id': f't{i}', 'repository': f'repo{i}', 'split': split}
             for i, split in enumerate(('train', 'dev', 'holdout', 'custom'))]
    registry.write_text(json.dumps({'tasks': tasks}), encoding='utf-8')
    digest = data.sha256(registry)
    trace = {'messages': [{'role': 'system', 'content': 'Use REPL'},
                         {'role': 'user', 'content': 'Repair this repository'}],
             'response': '```repl\nprint(context)\n```', 'input_tokens': 8, 'output_tokens': 8, 'kind': 'root'}
    report = {'schema_version': 1, 'split': 'train', 'tasks_sha256': digest,
              'tasks': [dict(t, passed=True, mode='rlm', trace=[trace, trace, dict(trace, kind='subcall')]) for t in tasks]}
    report_file = root / 'report.json'
    def write_report(value):
        report_file.write_text(json.dumps(value), encoding='utf-8')
    write_report(report)
    output = root / 'training.jsonl'
    manifest = data.export([report_file], registry, output)
    rows = data.load_verified(output, registry)
    assert len(rows) == 1 and rows[0]['task_id'] == 't0'
    assert rows[0]['messages'][-1]['content'] == trace['response']
    assert manifest['excluded']['split'] == 3 and manifest['excluded']['duplicate'] == 1
    assert manifest['excluded']['subcall'] == 1
    rejected(lambda: data.export([report_file], registry, output))
    for field, value in [('tasks_sha256', '0' * 64), ('schema_version', 9), ('split', 'holdout')]:
        bad = copy.deepcopy(report)
        bad[field] = value
        write_report(bad)
        rejected(lambda: data.export([report_file], registry, root / f'{field}.jsonl'))
    bad = copy.deepcopy(report)
    bad['tasks'][0]['repository'] = 'repo2'
    write_report(bad)
    rejected(lambda: data.export([report_file], registry, root / 'spoof.jsonl'))
    for change in ({'response': ''}, {'messages': []}, {'input_tokens': -1},
                   {'response': 'x' * (data.MAX_CHARS + 1)}):
        bad = copy.deepcopy(report)
        bad['tasks'][0]['trace'][0].update(change)
        write_report(bad)
        rejected(lambda: data.export([report_file], registry, root / 'malformed.jsonl'))
    output.write_text(output.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    rejected(lambda: data.load_verified(output, registry))
    # Even matching dataset hashes cannot relabel a holdout task as training.
    rows[0]['task_id'] = 't2'
    rows[0]['repository'] = 'repo2'
    output.write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
    manifest['dataset_sha256'] = data.sha256(output)
    data.manifest_path(output).write_text(json.dumps(manifest), encoding='utf-8')
    rejected(lambda: data.load_verified(output, registry))


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        text = ''.join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return [ord(character) for character in text]

messages = [{'role': 'user', 'content': 'first'}, {'role': 'assistant', 'content': 'old answer'},
            {'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]
encoded = train_adapter.encode_row(Tokenizer(), {'messages': messages}, 100)
assert encoded['labels'][-6:] == list(map(ord, 'answer'))
assert all(label == -100 for label in encoded['labels'][:-6])
assert train_adapter.encode_row(Tokenizer(), {'messages': messages}, 2) is None
class InconsistentTokenizer(Tokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        tokens = super().apply_chat_template(messages, tokenize, add_generation_prompt, **kwargs)
        return tokens if add_generation_prompt else [0] + tokens
rejected(lambda: train_adapter.encode_row(InconsistentTokenizer(), {'messages': messages}, 100))
rejected(lambda: train_adapter.validate_args(train_adapter.parser().parse_args([
    '--dataset', 'data.jsonl', '--model', 'Qwen/model', '--revision', 'main', '--output', 'adapter'])))
print('PASS: train-only export, registry integrity, deduplication, dataset hashes, and assistant masking')
