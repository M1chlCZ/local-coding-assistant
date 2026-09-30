"""Small functional pilot, not a substitute for a full coding benchmark."""
import argparse
import json
import re
import subprocess
import time
import threading
import urllib.request
import uuid
from pathlib import Path

IMAGE = "python@sha256:51dafde81dbdb6ebde285137a295cf18a47ca95234fe388a343719cb97305b3d"
CASES = [
    ("merge_intervals", "Write merge_intervals(intervals), returning sorted merged intervals as lists. Merge touching endpoints; accept unsorted input and an empty list; do not mutate the input.",
     "assert merge_intervals([])==[]\nx=[[5,7],[1,3],[3,5],[10,11]]\nassert merge_intervals(x)==[[1,7],[10,11]]\nassert x==[[5,7],[1,3],[3,5],[10,11]]\nassert merge_intervals([[1,1],[1,1]])==[[1,1]]"),
    ("parse_duration", "Write parse_duration(s), returning integer seconds. Accept only one or more groups of digits immediately followed by h, m, or s, in that order, each unit at most once. Reject invalid strings with ValueError. Examples: '1h30m', '0s', '45m', '2h3s'. No whitespace or signs.",
     "assert parse_duration('1h30m')==5400\nassert parse_duration('2h3s')==7203\nassert parse_duration('0s')==0\nfor x in ['', '3m2h','1s2s','1','-1s',' 1s','1hfoo','1h2']:\n try: parse_duration(x)\n except ValueError: pass\n else: raise AssertionError(x)"),
    ("top_k", "Write top_k(items, k). Return the k most frequent strings, highest frequency first, breaking ties lexicographically. Return [] for k<=0. If k exceeds the number of distinct strings, return all of them. Do not mutate items.",
     "assert top_k(['b','a','b','a','c'],2)==['a','b']\nassert top_k([],3)==[]\nassert top_k(['x'],0)==[]\nassert top_k(['b','a'],10)==['a','b']"),
    ("chunks", "Write chunks(items, size), returning a list of consecutive list chunks, preserving order. Reject size<=0 with ValueError. The last chunk may be shorter. Do not mutate input.",
     "assert chunks([1,2,3,4,5],2)==[[1,2],[3,4],[5]]\nassert chunks([],2)==[]\nfor n in [0,-2]:\n try: chunks([1],n)\n except ValueError: pass\n else: raise AssertionError(n)"),
]
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a project file.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "enum": ["solver.py", "test_solver.py"]}}, "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "write_file", "description": "Replace solver.py with corrected Python source.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "enum": ["solver.py"]}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the project's fixed tests.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]


def container_command(image, interactive=False):
    command = ["docker", "run", "--rm", "--network", "none", "--read-only",
               "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=32",
               "--memory=192m", "--cpus=1", "--user=65534:65534",
               "--tmpfs", "/workspace:rw,noexec,nosuid,size=4m,uid=65534,gid=65534"]
    return command + (["-i"] if interactive else ["-d"]) + [image]


def extract_code(content, preserve_indentation=False):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", content, re.S)
    code = match.group(1) if match else content
    return code.strip('\n') if preserve_indentation else code.strip()


def parse_tools(message):
    calls = message.get("tool_calls") or []
    if len(calls) > 4:
        raise ValueError("Too many tool calls in one response")
    for call in calls:
        function = call["function"]
        arguments = json.loads(function["arguments"])
        name = function["name"]
        if not isinstance(arguments, dict) or not isinstance(call.get("id"), str):
            raise ValueError("Malformed tool call")
        if name == "read_file":
            valid = set(arguments) == {"path"} and arguments["path"] in ["solver.py", "test_solver.py"]
        elif name == "write_file":
            valid = (set(arguments) == {"path", "content"} and arguments["path"] == "solver.py"
                     and isinstance(arguments["content"], str) and len(arguments["content"]) <= 100000)
        elif name == "run_tests":
            valid = not arguments
        else:
            valid = False
        if not valid:
            raise ValueError(f"Invalid arguments for {name}")
    return calls


def request(base, route, payload=None):
    req = urllib.request.Request(base + route, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def chat(base, messages, tools=None, max_tokens=768, thinking=False):
    payload = {"model": "local-coding-assistant", "messages": messages, "temperature": 0,
               "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": thinking}}
    if tools:
        payload["tools"] = tools
    started = time.monotonic()
    result = request(base, "/v1/chat/completions", payload)
    return result, round(time.monotonic() - started, 3)


def bounded_run(command, program, timeout=20):
    """Cap Docker client output on the host; callers own container cleanup."""
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    output = []

    def send():
        try:
            process.stdin.write(program.encode())
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def collect():
        data = process.stdout.read(4097)
        output.append(data)
        if len(data) > 4096:
            process.kill()

    writer = threading.Thread(target=send, daemon=True)
    reader = threading.Thread(target=collect, daemon=True)
    writer.start()
    reader.start()
    try:
        process.wait(timeout=timeout)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        writer.join(timeout=1)
        reader.join(timeout=1)
    data = output[0] if output else b''
    exceeded = len(data) > 4096
    return {'exit_code': process.returncode, 'output_limited': exceeded,
            'output': data[:4096].decode('utf-8', errors='replace') +
                      ('\nOutput limit exceeded.' if exceeded else '')}


def check_code(code, checks, timeout=20):
    program = code + "\n" + checks + "\nprint('PILOT_CHECKS_PASSED')\n"
    name = 'local-coding-check-' + uuid.uuid4().hex
    command = container_command(IMAGE, True)
    command[2:2] = ['--name', name]
    try:
        run = bounded_run(command + ["python", "-I", "-S", "-"], program, timeout)
        return {"passed": run['exit_code'] == 0 and not run['output_limited'] and "PILOT_CHECKS_PASSED" in run['output'],
                "exit_code": run['exit_code'], "feedback": run['output']}
    except subprocess.TimeoutExpired:
        return {"passed": False, "exit_code": None, "feedback": f"Execution timeout after {timeout} seconds"}
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=15)


def run_agent(base, thinking=False):
    initial = "def unique_sorted(values):\n    return sorted(values)\n"
    tests = ("import importlib.util\ns=importlib.util.spec_from_file_location('solver','/workspace/solver.py')\n"
             "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
             "assert m.unique_sorted([3,1,3,2])==[1,2,3]\nassert m.unique_sorted([])==[]\n"
             "x=[2,1,2];assert m.unique_sorted(x)==[1,2];assert x==[2,1,2]\nprint('PILOT_CHECKS_PASSED')\n")
    container = subprocess.check_output(container_command(IMAGE) + ["sleep", "600"], text=True).strip()
    trace = []

    def write(path, content):
        subprocess.run(["docker", "exec", "-i", container, "python", "-I", "-S", "-c",
                        "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text(sys.stdin.read())",
                        "/workspace/" + path], input=content, text=True, check=True, timeout=10)

    def test():
        def intact():
            return subprocess.check_output(["docker", "exec", container, "cat", "/workspace/test_solver.py"],
                                           text=True, timeout=10) == tests
        if not intact():
            return {"passed": False, "output": "Fixed test file was modified."}
        r = bounded_run(["docker", "exec", "-i", container, "python", "-I", "-S"], tests)
        if r['output_limited']:
            raise ValueError('Generated output exceeded limit; ending agent run')
        if not intact():
            return {"passed": False, "output": "Fixed test file was modified."}
        return {"passed": r['exit_code'] == 0 and "PILOT_CHECKS_PASSED" in r['output'],
                "output": r['output']}

    try:
        write("solver.py", initial)
        write("test_solver.py", tests)
        assert not test()["passed"], "Fixture must fail before the agent changes it"
        messages = [{"role": "system", "content": "You are an English coding assistant. Use the provided tools to inspect, edit, and test the repository. Repair failed tests before reporting success."},
                    {"role": "user", "content": "Fix unique_sorted in solver.py so it returns sorted unique values without changing the input. Inspect the files and run the tests."}]
        for step in range(6):
            result, elapsed = chat(base, messages, TOOLS, thinking=thinking)
            message = result["choices"][0]["message"]
            trace.append({"step": step, "message": message, "elapsed_s": elapsed, "usage": result.get("usage")})
            calls = parse_tools(message)
            messages.append({k: v for k, v in message.items() if k in ["role", "content", "tool_calls"]})
            if not calls:
                break
            for call in calls:
                function = call["function"]
                arguments = json.loads(function["arguments"])
                if function["name"] == "read_file":
                    output = subprocess.check_output(["docker", "exec", container, "cat", "/workspace/" + arguments["path"]], text=True, timeout=10)
                elif function["name"] == "write_file":
                    write(arguments["path"], arguments["content"])
                    output = "File saved. Run the tests to verify the repair."
                else:
                    output = json.dumps(test())
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        return {"name": "repository_agent", **test(), "trace": trace}
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18080")
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", type=int, default=8192, help="Context configured on the tested server")
    parser.add_argument("--label", required=True)
    parser.add_argument("--thinking", action="store_true", help="Enable model reasoning for this separate trial")
    args = parser.parse_args()
    report = {"label": args.label, "context": args.context, "thinking": args.thinking, "container_image": IMAGE,
              "scope": "Four Python functional smoke tasks, each with a first attempt and at most one repair using test feedback, plus one bounded repository agent task. Not a general coding benchmark.",
              "server_properties": request(args.base, "/props"), "results": []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, prompt, checks in CASES:
        row = {"name": name, "attempts": []}
        messages = [{"role": "system", "content": "You are an English coding assistant. Return only Python source code, including any needed standard-library imports."},
                    {"role": "user", "content": prompt}]
        try:
            for attempt in range(2):
                result, elapsed = chat(args.base, messages, thinking=args.thinking)
                content = result["choices"][0]["message"].get("content") or ""
                checked = check_code(extract_code(content), checks)
                row["attempts"].append({"elapsed_s": elapsed, "usage": result.get("usage"),
                                        "timings": result.get("timings"), "content": content, **checked})
                if checked["passed"]:
                    break
                messages += [{"role": "assistant", "content": content},
                             {"role": "user", "content": "The tests failed. Return corrected Python source only.\n" + checked["feedback"]}]
            row["passed"] = row["attempts"][-1]["passed"]
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            row.update(passed=False, error=str(error))
        report["results"].append(row)
        save()
        print(f"{name}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    try:
        report["results"].append(run_agent(args.base, thinking=args.thinking))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        report["results"].append({"name": "repository_agent", "passed": False, "error": str(error)})
    save()
    print(f"Pilot: {sum(r['passed'] for r in report['results'])}/{len(report['results'])} passed; {output}")


if __name__ == "__main__":
    main()
