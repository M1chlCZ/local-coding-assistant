"""Optional small actions inside the isolated REPL; no solutions or tests are supplied."""
import ast
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile

PROMPT = '''You repair repositories using a persistent isolated Python REPL.
Use executable ```repl``` blocks. Keep each step short and inspect its result before continuing.
task and context refer to the same already-loaded dictionary with task (requirements),
editable (paths), and files (original sources). No context file needs loading or JSON parsing.
is_child is True only inside a child REPL; it is False in the root.
The same interface exists in parent and child; never parse task or source text as JSON.
show() displays the requirements, inventory and editable source.
read(path) displays and returns source text. write(path, source) replaces ONE editable file after syntax validation.
check(code) runs your Python checks in a fresh process with repository imports. It prints and returns
returncode/output. Write actual assertions from requirements and examples; no tests are guaranteed.
Errors are evidence: correct the failing source or action, then rerun checks. Do not repeat unchanged failures.
finish() returns changed editable files from disk with automatic JSON serialization. It requires a
successful check on the current files. No JSON strings, manual escaping, or answer dictionary needed.
Do not modify task, supplied tests, supporting read-only files, or helpers. Hidden tests grade repairs.
Only your code can solve the task. Helpers do not supply fixes or checks.
'''
RECURSIVE = ('You are the root REPL. Use proposal = delegate("specific repair goal") after inspection. '
    'It sends the current task and files to a real child REPL, and returns its patch as a dictionary. '
    'Then apply(proposal), run your own independent check(code), and finish(). '
    'Verify required exceptions with try/except; an exception is raised, not returned as a value.\n')
CHILD = ('You are the child REPL. Inspect the supplied task and source, run a failing check, '
    'repair the files, rerun the check successfully and finish(). Do not delegate further. '
    'The parent will apply your returned patch and independently check it. '
    'Verify required exceptions with try/except; an exception is raised, not returned as a value.\n')


def task_context(context):
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except ValueError:
            context = dict(task=context, files={}, editable=[])
    if not isinstance(context, dict) or not isinstance(context.get('files'), dict):
        raise ValueError('Expected a repository task')
    return json.loads(json.dumps(context))


class Workspace:
    def __init__(self, context, answer, query=None, root='/workspace'):
        self.is_child = isinstance(context, str)
        self.task = task_context(context)
        self.answer, self.query = answer, query
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.checks = []
        for name, source in self.task['files'].items():
            target = self.target(name)
            if not isinstance(source, str):
                raise ValueError('Source must be text')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)

    def target(self, name):
        if (not isinstance(name, str) or not name or '\\' in name or ':' in name
                or PurePosixPath(name).is_absolute()
                or any(not p or p.startswith('.') for p in name.split('/'))):
            raise ValueError('Invalid repository path')
        target = self.root/name
        if target.resolve() != target or self.root not in target.resolve().parents:
            raise ValueError('Symlinks and paths outside the repository are not allowed')
        return target

    def _read(self, name):
        if name not in self.task['files']:
            raise ValueError('Unknown repository file')
        return self.target(name).read_text()

    # Keep legacy name= calls alongside the advertised path= keyword.
    def read(self, path=None, *, name=None):
        if path is not None and name is not None:
            raise ValueError('Provide one repository path')
        source = self._read(path if path is not None else name)
        print(source, end='')
        return source

    def show(self):
        print(self.task.get('task', '')[:3000])
        print('Files:', list(self.task['files']))
        print('Editable:', self.task['editable'])
        for name in self.task['editable'][:2]:
            print(name, self._read(name)[:1500])

    def validate(self, patch):
        if not isinstance(patch, dict) or not patch:
            raise ValueError('Expected a nonempty patch dictionary')
        for name, source in patch.items():
            if (name not in self.task['editable'] or name not in self.task['files']
                    or not isinstance(source, str) or len(source) > 100000):
                raise ValueError('Only editable source files may change')
            self.target(name)
            if name.endswith('.py'):
                compile(source, name, 'exec')
        return patch

    def write(self, path=None, source=None, *, name=None):
        if path is not None and name is not None:
            raise ValueError('Provide one repository path')
        self.apply({path if path is not None else name: source})

    def apply(self, patch):
        self.validate(patch)
        for name, source in patch.items():
            self.target(name).write_text(source)

    def hashes(self):
        return {p: hashlib.sha256(self.target(p).read_bytes()).hexdigest()
                for p in self.task['editable']}

    def check(self, code):
        if not isinstance(code, str) or len(code) > 12000:
            raise ValueError('Check code must be text of at most 12000 characters')
        tree = ast.parse(code)
        before = self.hashes()
        program = 'import sys;sys.path.insert(0,'+repr(str(self.root))+');exec('+repr(code)+')'
        with tempfile.TemporaryFile() as output:
            result = subprocess.run([sys.executable, '-B', '-I', '-c', program],
                cwd=self.root, stdout=output, stderr=subprocess.STDOUT, timeout=10)
            size = output.tell()
            output.seek(max(0, size-4096))
            text = output.read(4096).decode(errors='replace')
        receipt = dict(code=code, returncode=result.returncode, output=text,
            assertions=sum(isinstance(n, ast.Assert) for n in ast.walk(tree)),
            before=before, after=self.hashes())
        if len(json.dumps(self.checks+[receipt])) > 60000:
            raise RuntimeError('Cell output limit exceeded')
        self.checks.append(receipt)
        print('Check exit:', result.returncode)
        print(text)
        return receipt

    def delegate(self, goal):
        if self.is_child or not self.query or not isinstance(goal, str) or not goal or len(goal) > 2000:
            raise ValueError('Delegation needs an available child and a short goal')
        request = dict(task=self.task['task']+'\nChild goal: '+goal,
            editable=self.task['editable'], files={p:self._read(p) for p in self.task['files']})
        response = self.query(json.dumps(request))
        match = re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n```\s*', response, re.S)
        result = json.loads(match.group(1) if match else response)
        result = self.validate(result)
        print(f'Child returned a patch for {len(result)} editable file(s); source/path validation passed. '
              'Apply the returned dictionary and run your own check before finish().')
        return result

    def finish(self):
        patch = {p:self._read(p) for p in self.task['editable']
                 if self._read(p) != self.task['files'][p]}
        self.validate(patch)
        if (not self.checks or self.checks[-1]['returncode'] != 0
                or self.checks[-1]['before'] != self.hashes()
                or self.checks[-1]['after'] != self.hashes()):
            raise ValueError('Run a successful check on the current source before finishing')
        self.answer.update(content=patch, ready=True)

    def bindings(self):
        snapshot = json.loads(json.dumps(self.task))
        return dict(task=snapshot, context=snapshot, is_child=self.is_child, **{name:getattr(self, name)
                    for name in ('read', 'show', 'write', 'apply', 'check', 'delegate', 'finish')})
