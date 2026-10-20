"""Loopback-only BF16 Qwen3.5 baseline/adapter evaluator; never downloads weights."""
import argparse
import contextlib
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from qwen35_train import MODEL, load_model, read_json, sha256, validate_revision
from research.adapter_eval_server import completion, validate_batch
from structured_output import response_schema, check_response, StructuredOutput


def validate_adapter(path, model, revision):
    metadata = read_json(path / 'training.json')
    if (metadata.get('status') != 'completed' or metadata.get('model') != model
            or metadata.get('revision') != revision or metadata.get('precision') != 'bf16'
            or metadata.get('adapter_sha256') != sha256(path / 'adapter_model.safetensors')
            or metadata.get('adapter_config_sha256') != sha256(path / 'adapter_config.json')):
        raise ValueError('Only an intact, completed adapter for this pinned base can be evaluated')


def sampling(data):
    """Evaluation is greedy by default; collection can request reproducible sampling."""
    temperature = data.get('temperature', 0)
    seed = data.get('seed', 3407)
    if (type(temperature) not in (int, float) or not math.isfinite(temperature)
            or not 0 <= temperature <= 1 or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError('Use temperature 0 to 1 and a 32-bit unsigned seed')
    return ({'do_sample': True, 'temperature': temperature, 'top_p': .95}
            if temperature else {'do_sample': False}), seed


def stop_tokens(model, tokenizer):
    # The model config may end documents while the chat template ends assistant turns.
    configured = model.generation_config.eos_token_id
    endings = list(configured) if isinstance(configured, (list, tuple)) else [configured]
    if tokenizer.eos_token_id not in endings:
        endings.append(tokenizer.eos_token_id)
    endings = [token for token in endings if type(token) is int]
    if not endings:
        raise ValueError('Model and tokenizer have no generation stop token')
    return endings


def generate(model, tokenizer, data, use_adapter, has_adapter, output_limit, context, batch_size,
             structured=None):
    conversations, maximum = validate_batch(data, output_limit)
    schema = response_schema(data)
    if schema is not None and structured is None:
        raise ValueError('Start the server with --structured-output for structured requests')
    import torch
    options, seed = sampling(data)
    endings = stop_tokens(model, tokenizer)
    if options['do_sample']:
        torch.manual_seed(seed)
    responses = []
    # Small bounded batches protect the 16 GB BF16 working set, including long repair prompts.
    for start in range(0, len(conversations), batch_size):
        batch = conversations[start:start + batch_size]
        inputs = tokenizer.apply_chat_template(batch, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, padding=True, return_tensors='pt', return_dict=True)
        padded = inputs['input_ids'].shape[-1]
        if padded + maximum > context:
            raise ValueError(f'Input and output exceed the {context}-token context; no truncation applied')
        inputs = inputs.to('cuda')
        mode = model.disable_adapter() if has_adapter and not use_adapter else contextlib.nullcontext()
        constraints = {} if schema is None else {'logits_processor': [structured.processor(schema)]}
        with mode, torch.inference_mode():
            out = model.generate(**inputs, **options, **constraints, max_new_tokens=maximum,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=endings)
        responses.extend(completion(tokenizer, tokens[padded:], int(mask.sum()), maximum,
                                    endings)
                         for tokens, mask in zip(out.tolist(), inputs['attention_mask']))
    if schema is not None:
        for response in responses:
            check_response(response, schema)
    return responses


def make_handler(generate_fn, model_id, revision, has_adapter, structured_outputs=False):
    state = {'mode': 'base'}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                self.send_json({'error': 'Unknown path'}, 404)
                return
            self.send_json({'status': 'ok', 'model': model_id, 'revision': revision,
                            'precision': 'bf16', 'adapter': state['mode'] == 'adapter',
                            'adapter_available': has_adapter,
                            'structured_outputs': structured_outputs})

        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 2_000_000:
                    raise ValueError('Request size')
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError('Request requires an object')
                if self.path == '/mode':
                    mode = data.get('mode')
                    if mode not in ('base', 'adapter') or (mode == 'adapter' and not has_adapter):
                        raise ValueError('Requested model mode is unavailable')
                    state['mode'] = mode
                    self.send_json({'mode': mode})
                    return
                if self.path not in ('/batch', '/v1/chat/completions'):
                    self.send_json({'error': 'Unknown path'}, 404)
                    return
                if data.get('stream'):
                    raise ValueError('Streaming is not enabled for matched evaluation')
                batch = data if self.path == '/batch' else {**data, 'conversations': [data.get('messages')]}
                result = generate_fn(batch, state['mode'] == 'adapter')
                self.send_json({'responses': result} if self.path == '/batch' else result[0])
            except (ValueError, TypeError, KeyError) as error:
                self.send_json({'error': str(error)}, 400)
            except Exception as error:
                # Infrastructure errors must remain distinguishable from wrong coding answers.
                print(f'EVAL_ERROR {type(error).__name__}: {error}', flush=True)
                self.send_json({'error': f'{type(error).__name__}: {error}'}, 503)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--output-limit', type=int, choices=(1024, 2048), default=1024)
    parser.add_argument('--context-length', type=int, choices=(2048, 4096, 8192), default=8192)
    parser.add_argument('--batch-size', type=int, choices=(1, 2), default=1)
    parser.add_argument('--structured-output', action='store_true',
                        help='Enable optional XGrammar JSON constraints; never changes text requests')
    args = parser.parse_args()
    validate_revision(args.model, args.revision)
    if not 1024 <= args.port <= 65535:
        raise ValueError('Use a nonprivileged loopback port')
    if args.adapter:
        validate_adapter(args.adapter, args.model, args.revision)
    model, tokenizer, fast = load_model(args.model, args.revision, args.context_length)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    fast.for_inference(model)
    model.eval()
    model.config.use_cache = True
    structured = (StructuredOutput(tokenizer, model.get_output_embeddings().weight.shape[0],
                                  stop_tokens(model, tokenizer)) if args.structured_output else None)
    print('EVAL_STOP_TOKENS '+json.dumps(stop_tokens(model, tokenizer)), flush=True)
    def run(data, use_adapter):
        return generate(model, tokenizer, data, use_adapter, bool(args.adapter),
                        args.output_limit, args.context_length, args.batch_size, structured)
    server = HTTPServer(('127.0.0.1', args.port),
                        make_handler(run, args.model, args.revision, bool(args.adapter),
                                     args.structured_output))
    print(f'EVAL_READY http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
