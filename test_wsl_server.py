"""Verify shared server settings and safe Windows path conversion without loading a model."""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import wsl_server

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    executable = root/'runtime/llama/llama-server'
    executable.parent.mkdir(parents=True)
    executable.touch()
    (executable.parent/'libcublas.so.13').touch()
    model = root/'model.gguf'
    model.touch()
    for value in (str(model), 'C:\\models\\model.gguf'):
        with patch.object(sys, 'argv', ['wsl_server.py','--model',value,'--runtime',str(root/'runtime')]), \
             patch.object(wsl_server.subprocess, 'check_output', return_value=str(model)+'\n') as convert, \
             patch.object(os, 'execvpe') as execute:
            wsl_server.main()
            args = execute.call_args.args
            assert args[0] == str(executable.resolve())
            assert args[1][args[1].index('--host')+1] == '127.0.0.1'
            assert args[1][args[1].index('--n-cpu-ffn')+1] == '8'
            assert str(model) in args[1] and '/usr/lib/wsl/lib' in args[2]['LD_LIBRARY_PATH']
            if value.startswith('C:'):
                convert.assert_called_once_with(['wslpath','-u',value],text=True)
            else:
                convert.assert_not_called()
print('PASS: shared loopback CUDA settings and Windows path conversion')
