"""A fixed HumanEval subset, greedy first attempt only, executed in isolated containers."""
import argparse
import gzip
import json
import random
import re
import subprocess
import time
import textwrap
from pathlib import Path
from evaluation import chat, check_code, extract_code, request, IMAGE
from launcher import verify

ROOT = Path(__file__).resolve().parent

def assemble_solution(task, content):
    code = textwrap.dedent(extract_code(content, preserve_indentation=True))
    if re.search(r'(?m)^def\s+' + re.escape(task['entry_point']) + r'\b', code):
        return task['prompt'] + '\n' + code
    return task['prompt'] + textwrap.indent(code, '    ')

def load_tasks():
    metadata = json.loads((ROOT / 'research/humaneval_manifest.json').read_text())
    source = ROOT / 'research/HumanEval.jsonl.gz'
    verify(source, metadata)
    with gzip.open(source, 'rt', encoding='utf-8') as stream:
        tasks = [json.loads(line) for line in stream]
    return sorted(random.Random(metadata['subset_seed']).sample(tasks, metadata['subset_count']),
                  key=lambda t: int(t['task_id'].split('/')[1]))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:18080')
    parser.add_argument('--label', default='reference-solutions')
    parser.add_argument('--output', required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument("--thinking", action="store_true", help="Enable model reasoning for this separate trial")
    args = parser.parse_args()
    metadata = json.loads((ROOT / 'research/humaneval_manifest.json').read_text())
    report = {'label': args.label, 'dataset': metadata, 'container_image': IMAGE,
              'scope': '32 fixed HumanEval tasks, original tests, greedy pass@1; supplied code context retained and either full function or continuation accepted. Not the full suite or HumanEval+.',
              'max_generation_tokens': 2048, 'temperature': 0, 'thinking': args.thinking,
              'started_unix': time.time(), 'results': []}
    if not args.verify_only:
        report['server_properties'] = request(args.base, '/props')
    for task in load_tasks():
        row = {'task_id': task['task_id']}
        checks = task['test'] + '\ncheck(' + task['entry_point'] + ')'
        try:
            if args.verify_only:
                content = task['prompt'] + task['canonical_solution']
            else:
                result, elapsed = chat(args.base, [
                    {'role': 'system', 'content': 'You are an English coding assistant. Return complete, concise Python source only, including needed imports and supplied helper functions. Do not repeat docstrings, examples, or reasoning.'},
                    {'role': 'user', 'content': task['prompt']}], max_tokens=2048, thinking=args.thinking)
                content = result['choices'][0]['message'].get('content') or ''
                row.update(elapsed_s=elapsed, timings=result.get('timings'), usage=result.get('usage'),
                           finish_reason=result['choices'][0].get('finish_reason'))
            code = content if args.verify_only else assemble_solution(task, content)
            row.update(content=content, **check_code(code, checks))
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            row.update(passed=False, error=str(error))
        report['results'].append(row)
        report['ended_unix'] = time.time()
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f"{task['task_id']}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    passed = sum(row['passed'] for row in report['results'])
    print(f"{args.label}: {passed}/{len(report['results'])}", flush=True)
    if args.verify_only and passed != len(report['results']):
        raise SystemExit('Reference solutions must all pass before model results are trusted')

if __name__ == '__main__':
    main()
