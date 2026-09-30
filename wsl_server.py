"""Serve the existing model with the pinned Linux CUDA runtime inside WSL."""
import argparse
import os
import subprocess
from pathlib import Path, PureWindowsPath

from launcher import ROOT, server_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Existing GGUF path, Linux or Windows')
    parser.add_argument('--runtime', type=Path, default=ROOT/'.cache/runtime-wsl/b11146')
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--cpu-ffn', type=int, default=8)
    parser.add_argument('--cpu-moe', type=int, default=0)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--threads', type=int, default=12)
    args = parser.parse_args()
    model = args.model
    if PureWindowsPath(model).drive:
        model = subprocess.check_output(['wslpath', '-u', model], text=True).strip()
    executables = list(args.runtime.rglob('llama-server'))
    libraries = list(args.runtime.rglob('libcublas.so.*'))
    if len(executables) != 1 or not libraries:
        parser.error('Pinned Linux CUDA runtime is missing. See research/wsl.md.')
    executable = executables[0].resolve()
    command = server_command(executable, Path(model), args.context, args.cpu_moe,
                             args.port, args.threads, args.cpu_ffn)
    environment = os.environ.copy()
    paths = sorted({str(path.parent.resolve()) for path in libraries}) + ['/usr/lib/wsl/lib']
    if environment.get('LD_LIBRARY_PATH'):
        paths.append(environment['LD_LIBRARY_PATH'])
    environment['LD_LIBRARY_PATH'] = ':'.join(paths)
    os.execvpe(str(executable), command, environment)


if __name__ == '__main__':
    main()
