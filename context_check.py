"""Measure a near-full prompt and simple retrieval; this is not a coding benchmark."""
import argparse
import json
import time
import urllib.request
from pathlib import Path
from evaluation import request

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:18080')
    parser.add_argument('--context', required=True, type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 2048 <= args.context <= 262144:
        parser.error('Context must be between 2048 and 262144')
    head = 'Use this code reference to answer the question at the end.\nPROJECT_KEY = "oak-719-copper"\n'
    tail = '\nWhat is PROJECT_KEY? Return only its exact value.'
    lines = ['# Reference module %05d: def normalize_record(record): return dict(record)\n' % i
             for i in range(args.context // 10)]
    lo, hi = 0, len(lines)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        count = len(request(args.base, '/tokenize', {'content': head + ''.join(lines[:mid]) + tail})['tokens'])
        if count <= args.context - 1024:
            lo = mid
        else:
            hi = mid - 1
    prompt = head + ''.join(lines[:lo]) + tail
    count = len(request(args.base, '/tokenize', {'content': prompt})['tokens'])
    started = time.time()
    first_token = None
    result = {}
    content = ''
    payload = {'prompt': prompt, 'temperature': 0, 'n_predict': 64,
               'cache_prompt': False, 'stream': True}
    req = urllib.request.Request(args.base + '/completion', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            event = json.loads(line[6:])
            piece = event.get('content', '')
            if piece and first_token is None:
                first_token = time.time() - started
            content += piece
            if event.get('stop'):
                result = event
    result['content'] = content
    report = {'context': args.context, 'started_unix': started, 'ended_unix': time.time(),
              'prompt_tokens': count, 'first_token_s': first_token, 'elapsed_s': time.time() - started,
              'scope': 'Synthetic repeated code comments and one retrieval value; capacity/speed only.',
              'passed': 'oak-719-copper' in result.get('content', ''), 'response': result}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'response'}, indent=2), flush=True)
    print(result.get('timings'), flush=True)
    if not report['passed'] or first_token is None:
        raise SystemExit('Retrieval failed or no text was streamed')

if __name__ == '__main__':
    main()
