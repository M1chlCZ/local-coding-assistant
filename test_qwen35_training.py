"""CPU contract checks for verified data, exact recovery, and the loopback evaluator."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import qwen35_train as train
import qwen35_server as server


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return ''.join(m['role'] + ':' + m['content'] + '\n' for m in messages) + (
            'assistant:' if add_generation_prompt else '')

    def encode(self, text, add_special_tokens):
        return [ord(c) for c in text]


def row(task='repair', language='go'):
    return {'task_id': task, 'language': language, 'split': 'train',
            'messages': [{'role': 'user', 'content': 'Fix it'}, {'role': 'assistant', 'content': 'patch'}],
            'verification': {'kind': 'publisher_execution', 'fail_to_pass_count': 2,
                             'locally_replayed': False},
            'provenance': {'dataset': 'publisher/repairs', 'revision': 'a' * 40,
                           'source_row_sha256': 'b' * 64}}


def data_files(root, rows):
    data, manifest = root / 'train.jsonl', root / 'manifest.json'
    data.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    from collections import Counter
    train.atomic_json(manifest, {'schema_version': 1, 'split': 'train', 'rows': len(rows),
        'languages': dict(Counter(r['language'] for r in rows)), 'verification': 'publisher_execution',
        'dataset_sha256': train.sha256(data)})
    return data, manifest


def checkpoint(root, step, binding):
    path = root / f'checkpoint-{step}'
    path.mkdir()
    for name in train.CHECKPOINT_FILES:
        (path / name).write_text(json.dumps({'global_step': step}) if name == 'trainer_state.json' else name)
    train.complete_checkpoint(path, binding)
    return path


class TrainingTests(unittest.TestCase):
    def test_loader_preserves_pin_precision_and_native_text_architecture(self):
        from types import SimpleNamespace
        calls = []
        tokenizer = SimpleNamespace(get_vocab=lambda: range(100001), encode=lambda *a, **k: [1])
        model = SimpleNamespace(config=SimpleNamespace(model_type='qwen3_5_text'))
        def load(**kwargs):
            calls.append(kwargs)
            return model, tokenizer
        fake_fast = SimpleNamespace(from_pretrained=load)
        fake_torch = SimpleNamespace(bfloat16='bf16', cuda=SimpleNamespace(
            is_available=lambda: True, is_bf16_supported=lambda: True))
        with patch.dict('sys.modules', {'unsloth': SimpleNamespace(FastLanguageModel=fake_fast),
                                       'torch': fake_torch}):
            train.load_model(train.MODEL, 'a' * 40, 4096)
        config = calls[0]
        self.assertEqual(config['revision'], 'a' * 40)
        self.assertTrue(config['use_exact_model_name'])
        self.assertTrue(config['text_only'])
        self.assertTrue(config['local_files_only'])
        self.assertTrue(config['load_in_16bit'])
        self.assertFalse(config['load_in_4bit'])
        self.assertFalse(config['trust_remote_code'])

    def test_incomplete_tokenizer_cache_is_rejected(self):
        from types import SimpleNamespace
        incomplete = SimpleNamespace(get_vocab=lambda: {'<pad>': 0}, encode=lambda *a, **k: [])
        with self.assertRaisesRegex(ValueError, 'cache is incomplete'):
            train.validate_tokenizer(incomplete)

    def test_provenance_counts_holdout_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = [row(), row('ts', 'typescript')]
            data, manifest = data_files(root, valid)
            rows, metadata = train.load_verified(data, manifest)
            self.assertEqual(rows, valid)
            self.assertEqual(metadata['verification'], 'publisher_execution')
            for bad in [valid + [valid[0]], [dict(valid[0], split='test')],
                        [dict(valid[0], verification={'kind': 'publisher_execution', 'fail_to_pass_count': 0})],
                        [dict(valid[0], messages=['bad'])]]:
                data, manifest = data_files(root, bad)
                with self.assertRaises(ValueError):
                    train.load_verified(data, manifest)
            data, manifest = data_files(root, valid)
            data.write_text(data.read_text().replace('patch', 'tampered'))
            with self.assertRaises(ValueError):
                train.load_verified(data, manifest)

    def test_no_partial_answers_and_assistant_only_loss(self):
        short, long = row(), row('too-long', 'typescript')
        long['messages'][-1]['content'] = 'x' * 200
        encoded, report = train.encode_rows(Tokenizer(), [short, long], 100)
        self.assertEqual(report['included_count'], 1)
        self.assertEqual(report['excluded'][0]['task_id'], 'too-long')
        self.assertEqual(report['languages'], {'go': 1})
        labels = encoded[0]['labels']
        answer = [ord(c) for c in 'patch\n']
        self.assertEqual(labels[-len(answer):], answer)
        self.assertTrue(all(token == -100 for token in labels[:-len(answer)]))
        self.assertEqual(train.encode_rows(Tokenizer(), [short, long], 100)[1], report)

    def test_checkpoint_full_state_and_fail_closed_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = {'dataset': 'a', 'revision': 'b'}
            first = checkpoint(root, 2, binding)
            incomplete = root / 'checkpoint-3'
            incomplete.mkdir()
            (incomplete / 'adapter_model.safetensors').write_text('incomplete')
            self.assertEqual(train.latest_checkpoint(root, binding), first)
            with self.assertRaises(ValueError):
                train.latest_checkpoint(root, {'dataset': 'changed'})
            second = checkpoint(root, 4, binding)
            (second / 'rng_state.pth').write_text('corrupted')
            with self.assertRaises(ValueError):
                train.latest_checkpoint(root, binding)
            with self.assertRaises(FileNotFoundError):
                train.complete_checkpoint(incomplete, binding)
            self.assertFalse((incomplete / 'complete.json').exists())

    def test_resume_contract_and_preload_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, manifest = data_files(root, [row()])
            output = root / 'output'
            args = train.parser().parse_args(['--dataset', str(data), '--manifest', str(manifest),
                '--output', str(output), '--revision', 'a' * 40])
            train.validate_args(args)
            output.mkdir()
            args.resume = True
            with self.assertRaises(ValueError):
                train.validate_args(args)
            # A pre-model failure can restart, but is still bound to the original request.
            train.atomic_json(output / 'request.json', train.request_binding(args))
            train.validate_args(args)
            self.assertIsNone(train.latest_checkpoint(output, {}))
            original = train.request_binding(args)
            args.learning_rate *= 2
            self.assertNotEqual(original, train.request_binding(args))

    def test_adapter_requires_finished_bound_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'adapter_model.safetensors').write_text('weights')
            (root / 'adapter_config.json').write_text('{}')
            metadata = {'status': 'completed', 'model': train.MODEL, 'revision': 'a' * 40,
                'precision': 'bf16', 'adapter_sha256': train.sha256(root / 'adapter_model.safetensors'),
                'adapter_config_sha256': train.sha256(root / 'adapter_config.json')}
            train.atomic_json(root / 'training.json', metadata)
            server.validate_adapter(root, train.MODEL, 'a' * 40)
            with self.assertRaises(ValueError):
                server.validate_adapter(root, train.MODEL, 'b' * 40)
            train.atomic_json(root / 'training.json', {**metadata, 'status': 'paused'})
            with self.assertRaises(ValueError):
                server.validate_adapter(root, train.MODEL, 'a' * 40)


class ServerTests(unittest.TestCase):
    def test_modes_batch_and_infrastructure_failures(self):
        calls = []
        def generate(data, adapter):
            if data.get('raise_error'):
                raise RuntimeError('CUDA unavailable')
            conversations, maximum = server.validate_batch(data)
            calls.append((conversations, adapter, maximum))
            return [{'choices': [{'message': {'content': 'answer'}}]} for _ in conversations]
        http = HTTPServer(('127.0.0.1', 0), server.make_handler(generate, train.MODEL, 'a' * 40, False))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        def request(path, data=None):
            req = Request(f'http://127.0.0.1:{http.server_port}{path}',
                          data=json.dumps(data).encode() if data is not None else None,
                          headers={'Content-Type': 'application/json'})
            with urlopen(req, timeout=2) as response:
                return json.load(response)
        try:
            self.assertFalse(request('/health')['adapter_available'])
            with self.assertRaises(HTTPError) as failure:
                request('/mode', {'mode': 'adapter'})
            self.assertEqual(failure.exception.code, 400)
            self.assertEqual(request('/mode', {'mode': 'base'}), {'mode': 'base'})
            response = request('/batch', {'conversations': [[{'role': 'user', 'content': 'hi'}]]})
            self.assertEqual(len(response['responses']), 1)
            self.assertFalse(calls[-1][1])
            with self.assertRaises(HTTPError) as failure:
                request('/batch', {'raise_error': True})
            self.assertEqual(failure.exception.code, 503)
            with self.assertRaises(HTTPError) as failure:
                request('/v1/chat/completions', {'messages': [], 'stream': True})
            self.assertEqual(failure.exception.code, 400)
        finally:
            http.shutdown()
            thread.join()
            http.server_close()


if __name__ == '__main__':
    unittest.main()
