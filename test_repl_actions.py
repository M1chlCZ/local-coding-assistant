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

    def test_delegate_reports_returned_patch_without_echoing_source_or_inventing_test_success(self):
        patch = {'pkg/core.py': 'def greet():\n    return "SOURCE_SENTINEL"\n'}
        self.workspace.query = lambda _: json.dumps(patch)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(self.workspace.delegate('Repair and verify.'), patch)
        self.assertIn('Child returned a patch for 1 editable file', output.getvalue())
        self.assertIn('run your own check', output.getvalue())
        self.assertNotIn('SOURCE_SENTINEL', output.getvalue())
        self.assertNotIn('passed tests', output.getvalue())
        self.workspace.query = lambda _: json.dumps({'README.md': 'not editable'})
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(ValueError):
            self.workspace.delegate('Repair.')
        self.assertEqual(output.getvalue(), '')

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

    def test_advertised_path_keywords_and_legacy_names_work_in_parent_and_child(self):
        from recursive_agent import action_code_blocks
        child = Workspace(json.dumps(self.context), {}, root=self.root/'keyword-child')
        for workspace in (self.workspace, child):
            namespace = workspace.bindings()
            for argument in ("path='pkg/core.py'", "name='pkg/core.py'", "'pkg/core.py'"):
                with self.subTest(child=workspace.is_child, argument=argument):
                    source = 'def greet():\n    return "checked keyword repair"\n'
                    namespace['source'] = source
                    command = f'write({argument}, source=source)'
                    self.assertEqual(action_code_blocks(command), [command])
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(action_code_blocks(command)[0], namespace)
                        self.assertEqual(eval(f'read({argument})', namespace), source)
                        receipt = namespace['check']('from pkg.core import greet\nassert greet() == "checked keyword repair"')
                    self.assertEqual(receipt['returncode'], 0)
                    namespace['finish']()
                    self.assertEqual(workspace.answer['content'], {'pkg/core.py': source})

    def test_path_keyword_validation_rejects_invalid_or_ambiguous_edits_before_writing(self):
        old = self.workspace._read('pkg/core.py')
        for path, source in [('README.md', 'overwrite'), ('../escape.py', 'pass'),
                             ('pkg/core.py', 'def broken(:'), ('pkg/core.py', None)]:
            with self.subTest(path=path), self.assertRaises((ValueError, SyntaxError)):
                self.workspace.write(path=path, source=source)
        with self.assertRaises(ValueError):
            self.workspace.write(path='pkg/core.py', name='api.py', source='pass\n')
        with self.assertRaises(ValueError):
            self.workspace.read(path='pkg/core.py', name='api.py')
        self.assertEqual(self.workspace._read('pkg/core.py'), old)

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
        self.assertIn('Child returned a patch for 1 editable file', output.getvalue())
        self.assertNotIn(source, output.getvalue())
        self.assertEqual(self.answer['content'], {'pkg/core.py': source})

    @unittest.skipUnless(importlib.util.find_spec('rlm'), 'Pinned upstream and Docker required')
    def test_real_worker_displays_parent_and_child_reads_with_existing_output_bound(self):
        from recursive_agent import Sandbox
        context = dict(self.context, files={**self.context['files'], 'large.txt': 'x'*8193})
        for payload in (context, json.dumps(context)):
            env = Sandbox(context_payload=payload, action_api=True)
            try:
                result = env.execute_code('assert isinstance(context, dict)\nassert context is task\n'
                                          f'assert is_child is {isinstance(payload, str)}')
                self.assertFalse(result.stderr)
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

    def test_failed_checks_show_the_actual_assertion_line_in_parent_and_child(self):
        child = Workspace(json.dumps(self.context), {}, root=self.root/'feedback-child')
        code = ('from __future__ import annotations\nfrom pkg.core import greet\n'
                'observed = greet()\nassert observed == 1, "greeting mismatch"\n')
        for workspace in (self.workspace, child):
            with self.subTest(child=workspace.is_child), contextlib.redirect_stdout(io.StringIO()):
                workspace.write('pkg/core.py', 'def greet():\n    return 0\n')
                before = workspace.hashes()
                result = workspace.check(code)
                self.assertNotEqual(result['returncode'], 0)
                self.assertIn('\n    assert observed == 1, "greeting mismatch"\n', result['output'])
                self.assertIn('AssertionError: greeting mismatch', result['output'])
                self.assertEqual(result['code'], code)
                self.assertEqual(result['before'], before)
                self.assertEqual(result['after'], before)
                self.assertLessEqual(len(result['output'].encode()), 4096)
                with self.assertRaisesRegex(ValueError, 'successful check'):
                    workspace.finish()
                workspace.write('pkg/core.py', 'def greet():\n    return 1\n')
                self.assertEqual(workspace.check(code)['returncode'], 0)
                workspace.finish()

    def test_child_json_fence_uses_same_patch_format_as_root(self):
        proposal = {'pkg/core.py': 'def greet():\n    return 1\n'}
        for response in (json.dumps(proposal), '```json\n'+json.dumps(proposal)+'\n```'):
            self.workspace.query = lambda _, response=response: response
            self.assertEqual(self.workspace.delegate('Repair and test.'), proposal)
        for response in ('Here is a patch: '+json.dumps(proposal),
                         '```json\n{"README.md":"overwrite"}\n```'):
            self.workspace.query = lambda _, response=response: response
            with self.assertRaises(ValueError):
                self.workspace.delegate('Repair and test.')

    def test_execution_receipts_cannot_overflow_the_worker_protocol(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'Cell output limit exceeded'):
                for _ in range(6):
                    self.workspace.check('#'+'x'*11900+'\nassert True')
        self.assertLessEqual(len(json.dumps(self.workspace.checks)),60000)


if __name__ == '__main__':
    unittest.main()
