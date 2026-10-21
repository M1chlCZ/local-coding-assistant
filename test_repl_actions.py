"""The action interface transports text; it must never invent repairs or test results."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from repl_actions import Workspace


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = dict(task='Repair the greeting.', editable=['pkg/core.py', 'api.py'],
            files={'pkg/core.py': 'def greet():\n    return None\n',
                   'api.py': 'from pkg.core import greet\n', 'README.md': 'read only'})
        self.answer = {}
        self.workspace = Workspace(self.context, self.answer, root=self.root)

    def test_bare_read_displays_source_and_preserves_return_value_in_parent_and_child(self):
        child = Workspace(json.dumps(self.context), {}, root=self.root/'child')
        for workspace in (self.workspace, child):
            with self.subTest(child=workspace.is_child):
                namespace = workspace.bindings()
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec("read('README.md')", namespace)
                self.assertEqual(output.getvalue(), self.context['files']['README.md'])
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(namespace['read']('pkg/core.py'), self.context['files']['pkg/core.py'])

    def test_internal_reads_do_not_echo_source_during_show_delegate_or_finish(self):
        source = 'def greet():\n    return 2\n'
        self.workspace.query = lambda _: json.dumps({'pkg/core.py': source})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.workspace.show()
        self.assertEqual(output.getvalue().count(self.context['files']['pkg/core.py']), 1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.workspace.write('pkg/core.py', source)
            self.workspace.check('from pkg.core import greet\nassert greet() == 2')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.workspace.delegate('Repair and test.')
            self.workspace.finish()
        self.assertEqual(output.getvalue(), '')
        self.assertEqual(self.answer['content'], {'pkg/core.py': source})

    @unittest.skipUnless(importlib.util.find_spec('rlm'), 'Pinned upstream and Docker required')
    def test_real_worker_displays_parent_and_child_reads_with_existing_output_bound(self):
        from recursive_agent import Sandbox
        context = dict(self.context, files={**self.context['files'], 'large.txt': 'x'*8193})
        for payload in (context, json.dumps(context)):
            env = Sandbox(context_payload=payload, action_api=True)
            try:
                result = env.execute_code("read('README.md')")
                self.assertFalse(result.stderr)
                self.assertEqual(result.stdout, 'read only')
                result = env.execute_code("source = read('pkg/core.py')\nassert source == task['files']['pkg/core.py']")
                self.assertFalse(result.stderr)
                self.assertEqual(result.stdout, context['files']['pkg/core.py'])
                with self.assertRaisesRegex(RuntimeError, 'Cell output limit exceeded'):
                    env.execute_code("read('large.txt')")
            finally:
                env.cleanup()

    def test_text_roundtrip_and_fresh_checks(self):
        source = 'def greet():\n    return ' + repr('a "quote"\\path\nline') + '\n'
        self.workspace.write('pkg/core.py', source)
        self.assertEqual(self.workspace.read('pkg/core.py'), source)
        with self.assertRaisesRegex(ValueError, 'successful check'):
            self.workspace.finish()
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.workspace.check("from pkg.core import greet\nassert greet() == " + repr('a "quote"\\path\nline'))
        self.assertEqual(result['returncode'], 0)
        self.workspace.finish()
        self.assertEqual(json.loads(json.dumps(self.answer['content'])), {'pkg/core.py': source})
        self.workspace.write('pkg/core.py', 'def greet():\n    return 5\n')
        with self.assertRaisesRegex(ValueError, 'successful check'):
            self.workspace.finish()

    def test_invalid_edits_are_observable_and_do_not_damage_files(self):
        old = self.workspace.read('pkg/core.py')
        for path, source in [('README.md', 'overwrite'), ('../escape.py', 'pass'),
                             ('pkg/core.py', 'def broken(:'), ('api.py', 42)]:
            with self.assertRaises((ValueError, SyntaxError)):
                self.workspace.write(path, source)
        self.assertEqual(self.workspace.read('pkg/core.py'), old)
        with self.assertRaises(SyntaxError):
            self.workspace.apply({'pkg/core.py': 'pass\n', 'api.py': 'bad('})
        self.assertEqual(self.workspace.read('pkg/core.py'), old)
        (self.root/'api.py').unlink()
        (self.root/'api.py').symlink_to(self.root/'README.md')
        with self.assertRaises(ValueError):
            self.workspace.write('api.py', 'pass\n')

    def test_child_transport_and_failed_checks(self):
        sent = []
        proposal = {'pkg/core.py': 'def greet():\n    return 1\n'}
        def query(prompt):
            sent.append(json.loads(prompt))
            return json.dumps(proposal)
        self.workspace.query = query
        self.assertEqual(self.workspace.delegate('Repair and test the public API.'), proposal)
        self.assertEqual(sent[0]['files'], self.context['files'])
        self.assertEqual(self.workspace.read('pkg/core.py'), self.context['files']['pkg/core.py'])
        child = Workspace(json.dumps(sent[0]), {}, root=self.root/'child')
        self.assertEqual(child.read('README.md'), 'read only')
        with contextlib.redirect_stdout(io.StringIO()):
            result = child.check('from pkg.core import greet\nassert greet() == 1')
        self.assertNotEqual(result['returncode'], 0)
        self.assertEqual(result['assertions'], 1)
        self.assertIn('AssertionError', result['output'])
        self.assertEqual(len(child.checks), 1)
        self.workspace.query = lambda _: '{"unknown.py":"pass"}'
        with self.assertRaises(ValueError):
            self.workspace.delegate('repair')
        self.workspace.query = lambda _: 'invalid JSON'
        with self.assertRaises(ValueError):
            self.workspace.delegate('repair')

    def test_execution_receipts_cannot_overflow_the_worker_protocol(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'Cell output limit exceeded'):
                for _ in range(6):
                    self.workspace.check('#'+'x'*11900+'\nassert True')
        self.assertLessEqual(len(json.dumps(self.workspace.checks)),60000)


if __name__ == '__main__':
    unittest.main()
