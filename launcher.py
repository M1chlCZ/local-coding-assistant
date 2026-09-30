"""A dependency-free Windows launcher for the local coding-model experiment."""
import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
import zipfile
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"


def verify(path, artifact):
    if path.stat().st_size != artifact["size"]:
        raise RuntimeError(f"Incomplete artifact: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != artifact["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {path.name}; remove this file and run setup again")


def extract_runtime(archive, target):
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = member.filename.replace("\\", "/")
            destination = (target / name).resolve()
            if (PureWindowsPath(name).drive or not destination.is_relative_to(target.resolve())
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f"Unsafe archive entry: {member.filename}")
        z.extractall(target)


def server_command(exe, model, context, cpu_moe, port, threads):
    if not 512 <= context <= 262144:
        raise ValueError("Context must be between 512 and 262144 tokens")
    if not 0 <= cpu_moe <= 40:
        raise ValueError("CPU expert offload must be between 0 and 40 layers for this model")
    if not 1024 <= port <= 65535 or not 1 <= threads <= 128:
        raise ValueError("Use a port between 1024 and 65535 and 1-128 CPU threads")
    if not exe.is_file() or not model.is_file():
        raise RuntimeError("Runtime or model is missing. Run: python launcher.py setup")
    return [str(exe), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", str(context), "--parallel", "1", "--n-gpu-layers", "all",
            "--n-cpu-moe", str(cpu_moe), "--threads", str(threads),
            "--flash-attn", "on", "--log-verbosity", "4", "--jinja", "--no-mmproj", "--no-warmup",
            "--alias", "local-coding-assistant", "--reasoning", "off",
            "--repeat-penalty", "1.1", "--repeat-last-n", "256", "--ui-config",
            json.dumps({"systemMessage": "You are a coding assistant. Respond in English. "
                        "Write correct, simple code. State assumptions and distinguish verified results from suggestions."})]


def download(artifact, folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / artifact["name"]
    if path.name != artifact["name"] or PureWindowsPath(artifact["name"]).drive:
        raise ValueError("Artifact name must be a plain filename")
    if path.exists():
        verify(path, artifact)
        print(f"Verified: {path.name}", flush=True)
        return path
    if not artifact.get('url'):
        raise RuntimeError('This research model has no published download yet; build it with research/recipe.md')
    partial = path.with_name(path.name + ".partial")
    if partial.exists() and partial.stat().st_size > artifact["size"]:
        raise RuntimeError(f"Oversized partial download: {partial.name}; remove it and retry")
    print(f"Downloading {artifact['name']} ({artifact['size'] / 1e9:.2f} GB); resumable", flush=True)
    if not partial.exists() or partial.stat().st_size != artifact["size"]:
        curl = shutil.which("curl.exe") or shutil.which("curl")
        if not curl:
            raise RuntimeError("curl is required; it is included with current Windows installations")
        subprocess.run([curl, "--fail", "--location", "--no-progress-meter", "--retry", "4",
                        "--retry-all-errors", "--connect-timeout", "30", "--continue-at", "-",
                        "--output", str(partial), artifact["url"]], check=True)
    print(f"Checking SHA256: {partial.name}", flush=True)
    verify(partial, artifact)
    partial.replace(path)
    return path


def manifest(path=None):
    return json.loads((path or ROOT / "manifest.json").read_text(encoding="utf-8"))


def paths(config):
    runtime = CACHE / "runtime" / config["runtime"]["tag"]
    matches = list(runtime.rglob("llama-server.exe"))
    return (matches[0] if len(matches) == 1 else runtime / "llama-server.exe",
            CACHE / "models" / config["model"]["name"])


def setup(config, runtime_only=False):
    if sys.platform != "win32":
        raise RuntimeError("This first launcher targets Windows x64 with an NVIDIA GPU")
    if health(8080):
        raise RuntimeError('Stop the model server before setup; Windows locks runtime files while it runs')
    archives = [download(a, CACHE / "downloads") for a in config["runtime"]["artifacts"]]
    target = CACHE / "runtime" / config["runtime"]["tag"]
    for archive in archives:
        extract_runtime(archive, target)
    exe, _ = paths(config)
    subprocess.run([str(exe), "--version"], check=True)
    if not runtime_only:
        download(config["model"], CACHE / "models")
    print("Setup complete. Run: python launcher.py serve --open", flush=True)


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return json.load(response).get("status") == "ok"
    except (OSError, urllib.error.URLError):
        return False


def serve(config, args):
    exe, model = paths(config)
    defaults = config.get('defaults', {})
    for name, fallback in [('context', 8192), ('cpu_moe', 20), ('threads', 12)]:
        if getattr(args, name) is None:
            setattr(args, name, defaults.get(name, fallback))
    command = server_command(exe, model, args.context, args.cpu_moe, args.port, args.threads)
    if health(args.port):
        raise RuntimeError(f"A model server is already using port {args.port}")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"server-{time.time_ns()}.log"
    print(f"Model: {config['model']['label']}", flush=True)
    print(f"Context: {args.context}; CPU expert layers: {args.cpu_moe}; log: {log_path}", flush=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, cwd=exe.parent)
        try:
            deadline = time.monotonic() + 300
            while not health(args.port):
                if process.poll() is not None:
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
                    raise RuntimeError(f"Runtime exited ({process.returncode}).\n{tail}\n"
                                       "For memory errors, increase --cpu-moe or reduce --context.")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Startup took over 5 minutes; inspect {log_path}")
                time.sleep(1)
            url = f"http://127.0.0.1:{args.port}"
            print(f"Ready. Chat: {url} | Agent API: {url}/v1 | Ctrl+C to stop", flush=True)
            if args.open:
                webbrowser.open(url)
            process.wait()
            if process.returncode:
                raise RuntimeError(f"Runtime exited ({process.returncode}); inspect {log_path}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'manifest.json',
                        help='Model manifest; defaults to manifest.json')
    sub = parser.add_subparsers(dest="command", required=True)
    install = sub.add_parser("setup", help="Download and verify the runtime and selected model")
    install.add_argument("--runtime-only", action="store_true")
    start = sub.add_parser("serve", help="Start chat and a local API for agent harnesses")
    start.add_argument("--context", type=int, default=None)
    start.add_argument("--cpu-moe", type=int, default=None)
    start.add_argument("--threads", type=int, default=None)
    start.add_argument("--port", type=int, default=8080)
    start.add_argument("--open", action="store_true", help="Open the built-in chat in your browser")
    status = sub.add_parser("status", help="Show setup/download progress and server readiness")
    status.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    try:
        config = manifest(args.manifest)
        if args.command == "setup":
            setup(config, args.runtime_only)
        elif args.command == "serve":
            serve(config, args)
        else:
            exe, model = paths(config)
            partial = model.with_name(model.name + ".partial")
            downloaded = model.stat().st_size if model.exists() else partial.stat().st_size if partial.exists() else 0
            print(json.dumps({"runtime_installed": exe.is_file(), "model_present": model.is_file(),
                              "model_label": config["model"]["label"],
                              "downloaded_bytes": downloaded, "expected_bytes": config["model"]["size"],
                              "server_ready": health(args.port)}, indent=2))
    except KeyboardInterrupt:
        print("Stopped. Partial downloads can be resumed.")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
