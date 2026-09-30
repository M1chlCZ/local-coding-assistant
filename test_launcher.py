"""Run with python test_launcher.py; no dependencies or model download required."""
import hashlib
from argparse import Namespace
from unittest.mock import patch
import json
import tempfile
import zipfile
from pathlib import Path

try:
    import launcher
except ModuleNotFoundError:
    raise AssertionError("Launcher behavior has not been implemented yet") from None


def must_reject(action):
    try:
        action()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Unsafe input was accepted")


with tempfile.TemporaryDirectory(prefix="coding assistant ") as tmp:
    root = Path(tmp)
    exe, model = root / "llama-server.exe", root / "model with spaces.gguf"
    exe.touch()
    model.write_bytes(b"known model bytes")
    cmd = launcher.server_command(exe, model, 8192, 20, 8080, 12)
    assert cmd[0] == str(exe) and cmd[cmd.index("-m") + 1] == str(model)
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--n-cpu-moe") + 1] == "20"
    assert cmd[cmd.index('--repeat-penalty') + 1] == '1.1'
    assert cmd[cmd.index('--repeat-last-n') + 1] == '256'
    ui = json.loads(cmd[cmd.index("--ui-config") + 1])
    assert 'English' in ui['systemMessage'] and 'coding' in ui['systemMessage']
    for ctx, offload, port, threads in [(0, 20, 8080, 12), (8192, -1, 8080, 12),
                                       (8192, 41, 8080, 12), (8192, 20, 65536, 12),
                                       (8192, 20, 8080, 0)]:
        must_reject(lambda: launcher.server_command(exe, model, ctx, offload, port, threads))

    for defaults, override, expected in [({}, None, 20), ({'cpu_moe': 0}, None, 0),
                                          ({'cpu_moe': 4}, 7, 7)]:
        args = Namespace(context=None, cpu_moe=override, threads=None, port=8080)
        with patch.object(launcher, 'paths', return_value=(exe, model)), \
             patch.object(launcher, 'server_command', side_effect=RuntimeError('checked')) as build:
            must_reject(lambda: launcher.serve({'defaults': defaults}, args))
            assert build.call_args.args[2:] == (8192, expected, 8080, 12)

    with patch.object(launcher.sys, 'platform', 'win32'), patch.object(launcher, 'health', return_value=True):
        must_reject(lambda: launcher.setup({}))

    artifact = {"size": model.stat().st_size,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    launcher.verify(model, artifact)
    config = root / 'alternate.json'
    config.write_text('{"model": {"label": "local experiment"}}')
    assert launcher.manifest(config)['model']['label'] == 'local experiment'
    must_reject(lambda: launcher.download({'name': 'local.gguf', 'size': 1, 'sha256': '0'*64,
                                          'url': None}, root / 'downloads'))
    model.write_bytes(b"wrong model bytes")
    must_reject(lambda: launcher.verify(model, artifact))

    for unsafe in ["../escape.txt", "/absolute.txt", "C:\\escape.txt", "..\\escape.txt"]:
        archive = root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("ordinary.txt", "safe")
            z.writestr(unsafe, "unsafe")
        target = root / "runtime"
        must_reject(lambda: launcher.extract_runtime(archive, target))
        assert not target.exists(), "Extraction started before validating all entries"
    with zipfile.ZipFile(root / "safe.zip", "w") as z:
        z.writestr("bin/llama-server.exe", "known runtime")
    launcher.extract_runtime(root / "safe.zip", root / "runtime")
    assert (root / "runtime/bin/llama-server.exe").read_text() == "known runtime"

print("PASS: paths, settings, artifact integrity and safe runtime extraction")
