"""Opt-in output constraints; normal text generation has no optional dependencies."""
from functools import lru_cache
import json
import math
import re


def response_schema(data):
    if 'response_format' not in data:
        return None
    value = data['response_format']
    if not isinstance(value, dict):
        raise ValueError('response_format must be an object')
    if value == {'type': 'text'}:
        return None
    if value == {'type': 'json_object'}:
        return {'type': 'object'}
    if set(value) != {'type', 'json_schema'} or value['type'] != 'json_schema':
        raise ValueError('Supported response formats: text, json_object, json_schema')
    spec = value['json_schema']
    if (not isinstance(spec, dict) or set(spec) != {'name', 'strict', 'schema'}
            or spec['strict'] is not True or not isinstance(spec['name'], str)
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', spec['name'])):
        raise ValueError('json_schema requires name, strict: true, and schema')
    schema = spec['schema']
    try:
        encoded = json.dumps(schema, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError('Schema must be finite JSON') from error
    if len(encoded.encode()) > 16384 or not isinstance(schema, dict) or schema.get('type') != 'object':
        raise ValueError('Use an object schema of at most 16384 bytes')
    _validate_schema(schema)
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ValueError('Invalid JSON Schema') from error
    return schema


def _validate_schema(schema, depth=0):
    # Keep the supported contract explicit; never silently ignore unsupported constraints.
    kinds = {'object', 'array', 'string', 'integer', 'number', 'boolean', 'null'}
    if (depth > 16 or not isinstance(schema, dict) or not isinstance(schema.get('type'), str)
            or schema['type'] not in kinds):
        raise ValueError('Schemas require a supported type and at most 16 nesting levels')
    kind = schema['type']
    allowed = {'type', 'description', 'enum'}
    if kind == 'object':
        allowed |= {'properties', 'required', 'additionalProperties'}
    if kind == 'array':
        allowed |= {'items'}
    if set(schema) - allowed:
        raise ValueError('Unsupported schema keyword (references and formats are not supported)')
    if kind == 'object':
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        if (not isinstance(properties, dict) or not isinstance(required, list)
                or any(not isinstance(k, str) or k not in properties for k in required)
                or not isinstance(schema.get('additionalProperties', True), bool)):
            raise ValueError('Invalid object properties or required keys')
        for child in properties.values():
            _validate_schema(child, depth+1)
    elif kind == 'array':
        _validate_schema(schema.get('items'), depth+1)


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON key')
        value[key] = item
    return value


def _constant(value):
    raise ValueError('Non-finite JSON number')


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('JSON number exceeds finite float range')
    return number


def check_response(response, schema):
    """Keep raw evidence and finish reason, including syntactically complete truncations."""
    from jsonschema import Draft202012Validator
    choice = response['choices'][0]
    error = None
    if choice['finish_reason'] != 'stop':
        error = 'incomplete'
    else:
        try:
            value = json.loads(choice['message']['content'], object_pairs_hook=_object,
                               parse_constant=_constant, parse_float=_float)
        except (ValueError, TypeError, RecursionError):
            error = 'invalid_json'
        else:
            if not Draft202012Validator(schema).is_valid(value):
                error = 'schema_mismatch'
    response['structured_output'] = {'valid': error is None, 'error': error}


class StructuredOutput:
    def __init__(self, tokenizer, vocab_size, stop_tokens):
        import xgrammar as xgr
        info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=vocab_size, stop_token_ids=stop_tokens)
        self.compiler = xgr.GrammarCompiler(info, max_threads=2, cache_enabled=False)

    @lru_cache(maxsize=16)
    def compile(self, schema_json):
        # False preserves JSON Schema semantics for unspecified additional properties.
        return self.compiler.compile_json_schema(schema_json, strict_mode=False)

    def processor(self, schema):
        from xgrammar.contrib.hf import LogitsProcessor
        compiled = self.compile(json.dumps(schema, sort_keys=True, allow_nan=False))
        # Matchers contain generated-token state and must never survive a generate() call.
        return LogitsProcessor(compiled)
