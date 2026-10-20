"""Structured requests must fail explicitly and never turn truncation into success."""
import importlib
import json
import unittest
from unittest.mock import patch


PATCH = {
    'type': 'object',
    'properties': {'solution.py': {'type': 'string'}},
    'required': ['solution.py'],
    'additionalProperties': False,
}


def request(schema=PATCH):
    return {'response_format': {'type': 'json_schema', 'json_schema': {
        'name': 'patch', 'strict': True, 'schema': schema}}}


def response(text, reason='stop'):
    return {'choices': [{'message': {'content': text}, 'finish_reason': reason}]}


class StructuredOutputTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('structured_output'),
                             'Schema-constrained output support is missing')
        return importlib.import_module('structured_output')

    def test_text_is_unchanged_and_schema_is_not_mutated(self):
        module = self.module()
        for data in ({}, {'response_format': {'type': 'text'}}):
            self.assertIsNone(module.response_schema(data))
        data = request()
        before = json.dumps(data)
        self.assertEqual(module.response_schema(data), PATCH)
        self.assertEqual(json.dumps(data), before)
        self.assertEqual(module.response_schema({'response_format': {'type': 'json_object'}}),
                         {'type': 'object'})

    def test_unsupported_or_malformed_contracts_are_rejected(self):
        module = self.module()
        invalid = [None, [], {}, {'type': 'regex'}, {'type': 'text', 'schema': {}},
                   {'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': PATCH}},
                   {'type': 'json_schema', 'json_schema': {'name': 'x', 'strict': 1, 'schema': PATCH}}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.response_schema({'response_format': value})
        schemas = [
            {'type': 'object', '$ref': 'https://example.com/schema'},
            {'type': 'object', '$ref': '#/$defs/x', '$defs': {}},
            {'type': 'object', 'properties': {'x': {'type': 'string', 'format': 'email'}}},
            {'type': 'object', 'properties': {'x': {'type': 'unknown'}}},
            {'type': 'array', 'items': {'type': 'string'}},
            {'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['missing']},
            {'type': 'object', 'description': 'x' * 17000},
        ]
        for schema in schemas:
            with self.subTest(schema=str(schema)[:100]), self.assertRaises(ValueError):
                module.response_schema(request(schema))

    def test_valid_escaped_source_and_invalid_complete_or_truncated_results(self):
        module = self.module()
        source = 'def value():\n    return "Žluťoučký \\"kůň\\"\\\\path"\n'
        good = response(json.dumps({'solution.py': source}, ensure_ascii=False))
        module.check_response(good, PATCH)
        self.assertTrue(good['structured_output']['valid'])
        for text, reason in [
            ('{"solution.py": "unfinished', 'length'),
            (json.dumps({'solution.py': source}), 'length'),
            ('{"solution.py": 42}', 'stop'),
            ('{"other.py": "pass"}', 'stop'),
            ('{"solution.py":"a","solution.py":"b"}', 'stop'),
            ('{"solution.py":NaN}', 'stop'),
            ('```json\n{"solution.py":"pass"}\n```', 'stop'),
        ]:
            with self.subTest(text=text):
                value = response(text, reason)
                module.check_response(value, PATCH)
                self.assertFalse(value['structured_output']['valid'])
                self.assertEqual(value['choices'][0]['message']['content'], text)
                self.assertEqual(value['choices'][0]['finish_reason'], reason)
        huge_number = response('{"value":1e9999}')
        module.check_response(huge_number, {'type': 'object'})
        self.assertFalse(huge_number['structured_output']['valid'])

    def test_requested_constraint_cannot_silently_run_without_engine(self):
        self.module()
        import qwen35_server
        data = {**request(), 'conversations': [[{'role': 'user', 'content': 'repair'}]]}
        with self.assertRaisesRegex(ValueError, 'structured'):
            qwen35_server.generate(None, None, data, False, False, 1024, 8192, 1)

    def test_generation_uses_fresh_matchers_per_batch_and_leaves_text_unchanged(self):
        import contextlib
        import sys
        from types import SimpleNamespace
        import qwen35_server
        processors = []
        class Engine:
            def processor(self, schema):
                self.assert_schema = schema == PATCH
                value = object()
                processors.append(value)
                return value
        class Inputs(dict):
            def to(self, device):
                return self
        class Tokenizer:
            eos_token_id, pad_token_id = 99, 99
            def apply_chat_template(self, batch, **kwargs):
                return Inputs(input_ids=SimpleNamespace(shape=(len(batch), 2)),
                              attention_mask=[SimpleNamespace(sum=lambda: 2) for _ in batch])
            def decode(self, tokens, **kwargs):
                return '{"solution.py":"pass"}'
        received = []
        class Model:
            generation_config = SimpleNamespace(eos_token_id=[98, 99])
            def generate(self, input_ids, **kwargs):
                received.append(kwargs)
                return SimpleNamespace(tolist=lambda: [[1, 2, 3, 99]] * input_ids.shape[0])
        data = {**request(), 'conversations': [[{'role':'user', 'content': str(i)}] for i in range(3)]}
        with patch.dict(sys.modules, {'torch': SimpleNamespace(inference_mode=contextlib.nullcontext)}):
            result = qwen35_server.generate(Model(), Tokenizer(), data, False, False, 1024, 8192, 2, Engine())
            plain = qwen35_server.generate(Model(), Tokenizer(), {'conversations': data['conversations'][:1]},
                                          False, False, 1024, 8192, 1, Engine())
        self.assertEqual(len(processors), 2)
        self.assertIsNot(processors[0], processors[1])
        self.assertEqual([r['logits_processor'][0] for r in received[:2]], processors)
        self.assertNotIn('logits_processor', received[-1])
        self.assertNotIn('structured_output', plain[0])
        self.assertTrue(all(r['structured_output']['valid'] for r in result))


@unittest.skipUnless(importlib.util.find_spec('xgrammar'), 'Optional XGrammar environment required')
class GrammarIntegrationTests(unittest.TestCase):
    def test_actual_cpu_masks_escape_roundtrip_and_fresh_matchers(self):
        import torch
        import xgrammar as xgr
        from structured_output import StructuredOutput
        # Include a multi-character token, padded vocabulary entries and two EOS tokens.
        vocab = [chr(i) for i in range(128)] + ['<end>', '<turn>', '<forbidden>']
        info = xgr.TokenizerInfo(vocab, stop_token_ids=[128, 129], vocab_size=136)
        engine = StructuredOutput.__new__(StructuredOutput)
        engine.compiler = xgr.GrammarCompiler(info, max_threads=2, cache_enabled=False)
        text = json.dumps({'solution.py': 'x = "kůň\\\\path"\n'}, ensure_ascii=True)
        for _ in range(2):
            processor = engine.processor(PATCH)
            tokens = [0, 0]  # Prefix is outside the generated grammar.
            for character in text:
                scores = processor(torch.tensor([tokens]), torch.zeros(1, 136))
                self.assertTrue(torch.isfinite(scores[0, ord(character)]), repr(character))
                self.assertFalse(torch.isfinite(scores[0, 135]))
                self.assertFalse(torch.isfinite(scores[0, 128]))
                self.assertFalse(torch.isfinite(scores[0, 129]))
                tokens.append(ord(character))
            scores = processor(torch.tensor([tokens]), torch.zeros(1, 136))
            self.assertTrue(torch.isfinite(scores[0, 128]))
            self.assertTrue(torch.isfinite(scores[0, 129]))
        start = engine.processor(PATCH)(torch.tensor([[0]]), torch.zeros(1, 136))
        self.assertFalse(torch.isfinite(start[0, ord('}')]))
        self.assertFalse(torch.isfinite(start[0, 128]))
        self.assertFalse(torch.isfinite(start[0, 130]))
        self.assertEqual(engine.compile.cache_info().currsize, 1)


if __name__ == '__main__':
    unittest.main()
