"""Bounded recovery curriculum with explicit, balanced multilingual replay."""
import argparse
import json
from collections import Counter
from pathlib import Path

import agent_curriculum as curriculum
import agent_multifile as multi
from qwen35_train import atomic_json, load_verified, sha256


def balanced_rows(actions, anchors):
    languages = ('python', 'go', 'typescript', 'rust', 'dart')
    pools = {language: [r for r in anchors if r['language'] == language] for language in languages}
    if (not actions or any(r.get('example_kind') != 'executed_repl_repair' for r in actions)
            or len(anchors) != 150 or any(len(p) != 30 for p in pools.values())
            or len({r['task_id'] for r in actions + anchors}) != len(actions) + len(anchors)):
        raise ValueError('Require complete executed actions and 150 distinct verified replay examples')
    replay = []
    # Repetition changes sampling weight, not the number of independently verified examples.
    for index in range((len(actions) + 4) // 5):
        for language in languages:
            source = pools[language][index % 30]
            replay.append(dict(source, task_id=f"balanced-replay-{language}-{index}",
                               replay_source_id=source['task_id']))
    return actions + replay


def prepare(folder, source, evaluation):
    multi.prepare(folder, source, evaluation)
    values = json.loads((folder/'tasks.json').read_text())
    selected = sorted(values, key=lambda value: value['repository'])[:64]
    if len(selected) != 64:
        raise ValueError('Need 64 distinct verified training families')
    prepared = json.loads((folder/'prepare.json').read_text())
    atomic_json(folder/'selection.json', dict(eligible_tasks_sha256=sha256(folder/'tasks.json'),
        eligible_families=len(values), selected=[v['repository'] for v in selected],
        rule='First 64 valid families in repository-name order, before any model evaluation'))
    atomic_json(folder/'tasks.json', selected)
    atomic_json(folder/'prepare.json', dict(prepared, families=len(selected),
        tasks_sha256=sha256(folder/'tasks.json'), selection_sha256=sha256(folder/'selection.json')))


def balance(folder, anchors, revision):
    from transformers import AutoTokenizer
    from qwen35_train import MODEL, encode_rows, validate_tokenizer
    rows, metadata = load_verified(folder/'train.jsonl', folder/'train.jsonl.manifest.json')
    replay = curriculum.anchors(anchors)
    if rows[-150:] != replay:
        raise ValueError('Unexpected source replay; preserve original collection evidence')
    previous = json.loads((folder/'feasibility.json').read_text())
    mixed = balanced_rows(rows[:-150], replay)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision,
        local_files_only=True, trust_remote_code=False)
    validate_tokenizer(tokenizer)
    _, counts = encode_rows(tokenizer, mixed, 4096)
    if counts['excluded_count']:
        raise ValueError('Balanced dataset would truncate a complete training example')
    mix = dict(action_rows=len(rows)-150, replay_rows=len(mixed)-len(rows)+150,
        distinct_replay_examples=150, replay_languages=dict(Counter(r['language'] for r in mixed if 'replay_source_id' in r)),
        original_dataset_sha256=sha256(folder/'train.jsonl'), anchors_sha256=sha256(anchors),
        balance_unit='rows, not target tokens', oversampled=True)
    atomic_json(folder/'mixture.json', mix)
    curriculum.write_dataset(folder, mixed, dict(metadata['binding'], mixture=mix))
    atomic_json(folder/'feasibility.json', dict(previous, included_count=len(mixed),
        dataset_sha256=sha256(folder/'train.jsonl'), manifest_sha256=sha256(folder/'train.jsonl.manifest.json'),
        mixture_sha256=sha256(folder/'mixture.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'balance'))
    for name in ('folder', 'source', 'evaluation', 'anchors'):
        parser.add_argument('--'+name, type=Path, required=name == 'folder')
    parser.add_argument('--revision')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.folder, args.source, args.evaluation)
    else:
        balance(args.folder, args.anchors, args.revision)
