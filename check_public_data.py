"""Reject personal home paths and non-example SSH tunnel hosts before publication."""
import argparse
from pathlib import Path
import re
import subprocess


HOME = re.compile(rb'''(?i)(?:[a-z]:[\\/]+Users[\\/]+[a-z0-9_][^\\/\s"']*|/mnt/[a-z]/Users/[a-z0-9_][^/\s"']*|/Users/[a-z0-9_][^/\s"']*|/home/(?!coder(?:/|\b))[a-z0-9_.-]+/)''')
TUNNEL = re.compile(rb'(?m)^ssh[ \t]+-N[^\r\n]*')


def findings(content):
    matches = [(match.start(), 'personal home path') for match in HOME.finditer(content)]
    matches += [(match.start(), 'non-example SSH tunnel host') for match in TUNNEL.finditer(content)
                if not match.group().rstrip().endswith(b' coding-pc')]
    return [(content.count(b'\n', 0, offset) + 1, kind) for offset, kind in matches]


def check(history=False):
    if history:
        rows = subprocess.check_output(['git', 'rev-list', '--objects', '--all']).splitlines()
        objects = [(row.partition(b' ')[0], row.partition(b' ')[2]) for row in rows]
    else:
        rows = subprocess.check_output(['git', 'ls-tree', '-rz', 'HEAD']).split(b'\0')
        objects = [(row.partition(b'\t')[0].split()[2], row.partition(b'\t')[2]) for row in rows if row]
    failed = False
    for oid, path in objects:
        kind = subprocess.check_output(['git', 'cat-file', '-t', oid]).strip()
        if kind not in (b'blob', b'commit', b'tag'):
            continue
        content = subprocess.check_output(['git', 'cat-file', kind.decode(), oid])
        if b'\0' in content:
            continue
        for line, reason in findings(content):
            print(f'{path.decode(errors="replace") or oid.decode()}:{line}: {reason}')
            failed = True
    return failed


def self_test():
    user = b'example-user'
    for path in [b'C:' + b'\\Users\\' + user + b'\\project',
                 b'C:' + b'\\\\Users\\\\' + user + b'\\\\project',
                 b'/mnt/c/Users/' + user + b'/project', b'/Users/' + user + b'/project',
                 b'/home/' + user + b'/project']:
        assert findings(path), 'Personal home path was accepted'
    assert findings(b'ssh -N -L 127.0.0.1:18080:127.0.0.1:8080 ' + b'private-host')
    assert not findings(b'.cache/models/model.gguf\n/home/coder/local-coding-assistant')
    assert not findings(b'ssh -N -L 127.0.0.1:18080:127.0.0.1:8080 coding-pc')
    assert not findings(Path(__file__).read_bytes()), 'Checker source was mistaken for a personal path'
    print('PASS: path variants and SSH tunnel hosts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true', help='Check all reachable commits, tags, and file versions')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        raise SystemExit(1 if check(args.history) else 0)
