import json

import httpx
from pydantic import BaseModel, Field
import pytest

from toeic_engine.models import Review, Unit
from toeic_engine.openai_provider import OPENAI_MODEL, OpenAIProvider, openai_schema
from toeic_engine.provider import OutputError, ProviderError


RAW = json.dumps({'verdict': 'pass', 'naturalness': 5, 'toeic_alignment': 5,
                  'explanation_quality': 5, 'findings': [], 'item_reviews': []})


def stream(raw=RAW, *, model=OPENAI_MODEL, status='completed', terminal=True,
           refusal=False, final_text=None):
    events = [
        {'type': 'response.created', 'response': {'id': 'resp_test', 'model': model}},
        {'type': 'response.reasoning_text.delta', 'delta': 'PRIVATE NEVER SAVE'},
        {'type': 'response.output_text.delta', 'delta': raw},
    ]
    if refusal:
        events.append({'type': 'response.refusal.delta', 'delta': 'Refused'})
    if terminal:
        content = [{'type': 'output_text', 'text': raw if final_text is None else final_text}]
        if refusal:
            content.append({'type': 'refusal', 'refusal': 'Refused'})
        events.append({'type': f'response.{status}', 'response': {
            'id': 'resp_test', 'model': model, 'status': status,
            'output': [{'type': 'reasoning', 'summary': [{'text': 'PRIVATE NEVER SAVE'}]},
                       {'type': 'message', 'role': 'assistant', 'content': content}],
            'usage': {'input_tokens': 10, 'output_tokens': 20,
                      'output_tokens_details': {'reasoning_tokens': 8}},
        }})
    return ''.join('event: ' + e['type'] + '\ndata: ' + json.dumps(e) + '\n\n' for e in events)


def provider(tmp_path, monkeypatch, handler, max_calls=1):
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-private-openai-key')
    client = httpx.Client(base_url='https://api.openai.com', transport=httpx.MockTransport(handler))
    return OpenAIProvider(tmp_path, max_calls=max_calls, client=client)


def test_exact_model_effort_isolation_and_visible_only(tmp_path, monkeypatch):
    calls = []
    def handler(request):
        assert request.method == 'POST' and request.url.path == '/v1/responses'
        calls.append(json.loads(request.content))
        return httpx.Response(200, text=stream(), headers={'x-request-id': 'req_test'})
    p = provider(tmp_path, monkeypatch, handler)
    assert p.generate('the system', {'question': 'frozen'}, Review, 'r', max_tokens=8192).passes()
    body = calls[0]
    assert body['model'] == OPENAI_MODEL
    assert body['reasoning'] == {'effort': 'xhigh'}
    assert body['max_output_tokens'] == 65536
    assert body['stream'] is True and body['store'] is False
    assert body['instructions'] == 'the system'
    assert body['input'] == [{'role': 'user', 'content': '{"question": "frozen"}'}]
    assert body['text']['format']['type'] == 'json_schema'
    assert body['text']['format']['strict'] is True
    assert all(k not in body for k in ('previous_response_id', 'conversation', 'tools', 'temperature', 'fallbacks'))
    saved = ''.join(f.read_text() for f in tmp_path.iterdir())
    assert 'PRIVATE NEVER SAVE' not in saved and 'fake-private-openai-key' not in saved
    assert p.usage[0]['usage']['output_tokens_details']['reasoning_tokens'] == 8
    assert p.usage[0]['request_id'] == 'req_test'
    assert p.author_provenance == {'model': OPENAI_MODEL, 'effort': 'xhigh'}
    with pytest.raises(ProviderError, match='limit'):
        p.generate('x', {}, Review, 'r2')
    assert len(calls) == 1


@pytest.mark.parametrize('kwargs', [{'status': 'incomplete'}, {'terminal': False}, {'refusal': True},
                                   {'final_text': RAW + ' '}, {'raw': ''}])
def test_noncomplete_or_refusal_never_counts_as_pass(tmp_path, monkeypatch, kwargs):
    p = provider(tmp_path, monkeypatch, lambda _: httpx.Response(200, text=stream(**kwargs)))
    with pytest.raises(OutputError):
        p.generate('x', {}, Review, 'r')


def test_unrequested_model_rejected(tmp_path, monkeypatch):
    p = provider(tmp_path, monkeypatch, lambda _: httpx.Response(200, text=stream(model='gpt-6-astra')))
    with pytest.raises(ProviderError, match='Unexpected returned model'):
        p.generate('x', {}, Review, 'r')


def test_local_validation_not_relaxed(tmp_path, monkeypatch):
    raw = RAW.replace('"naturalness": 5', '"naturalness": 8')
    p = provider(tmp_path, monkeypatch, lambda _: httpx.Response(200, text=stream(raw)))
    with pytest.raises(OutputError):
        p.generate('x', {}, Review, 'r')


def test_strict_wire_schema_preserves_fields_and_full_local_schema():
    class Nested(BaseModel):
        title: str = 'heading'
        minimum: int = Field(default=1, ge=1)
        optional: str | None = None
    class Container(BaseModel):
        child: Nested
        rows: list[str] = Field(default_factory=list, min_length=2)
    schema = openai_schema(Container)
    nested = schema['$defs']['Nested']
    assert nested['required'] == ['title', 'minimum', 'optional']
    assert nested['additionalProperties'] is False
    assert nested['properties']['minimum'] == {'type': 'integer'}
    assert nested['properties']['optional']['anyOf'] == [{'type': 'string'}, {'type': 'null'}]
    assert schema['required'] == ['child', 'rows']
    assert schema['additionalProperties'] is False
    assert 'minItems' not in schema['properties']['rows']
    assert Container.model_json_schema()['properties']['rows']['minItems'] == 2
    unit = openai_schema(Unit)
    block = unit['$defs']['Block']
    assert set(block['required']) == set(block['properties'])
    assert {'text', 'speaker', 'timestamp', 'columns', 'rows'} <= set(block['required'])


def test_auth_error_redacts_all_provider_keys_and_does_not_retry(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-private-anthropic-key')
    monkeypatch.setenv('GOOGLE_CLOUD_API_KEY', 'fake-private-google-key')
    message = 'invalid fake-private-openai-key fake-private-anthropic-key fake-private-google-key'
    p = provider(tmp_path, monkeypatch, lambda _: httpx.Response(401, text=message), max_calls=3)
    with pytest.raises(ProviderError) as exc:
        p.generate('x', {}, Review, 'r')
    assert 'fake-private-' not in str(exc.value)
    assert 'fake-private-' not in (tmp_path / 'usage.json').read_text()
    assert p.calls == 1


def test_temporary_errors_retry_and_count_attempts(tmp_path, monkeypatch):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(429, text='rate limited') if len(seen) == 1 else httpx.Response(200, text=stream())
    monkeypatch.setattr('toeic_engine.openai_provider.time.sleep', lambda _: None)
    p = provider(tmp_path, monkeypatch, handler, max_calls=3)
    assert p.generate('x', {}, Review, 'r').passes()
    assert p.calls == 2 and [u['attempt'] for u in p.usage] == [1, 2]
    assert p.usage[0]['billing_complete'] is False


@pytest.mark.parametrize('error', [
    {'type': 'server_error', 'message': 'Temporary backend failure', 'param': None},
    {'code': 'rate_limit_exceeded', 'message': 'Temporary rate limit', 'param': 'model'},
])
def test_nested_transient_sse_error_retries_with_diagnostics(tmp_path, monkeypatch, error):
    seen = []
    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            event = {'type': 'error', 'error': error, 'debug': 'PRIVATE NEVER SAVE'}
            return httpx.Response(200, text='data: ' + json.dumps(event) + '\n\n')
        return httpx.Response(200, text=stream())
    monkeypatch.setattr('toeic_engine.openai_provider.time.sleep', lambda _: None)
    p = provider(tmp_path, monkeypatch, handler, max_calls=3)
    assert p.generate('x', {}, Review, 'r').passes()
    assert p.calls == 2
    assert p.usage[0]['api_error'] == {'type': 'error', 'error': error}
    assert error['message'] in p.usage[0]['detail']
    assert 'PRIVATE NEVER SAVE' not in (tmp_path / 'usage.json').read_text()


@pytest.mark.parametrize('error_type,error_code', [
    ('server_error', 'insufficient_quota'),
    ('authentication_error', 'invalid_api_key'),
    ('unrecognized_error', None),
])
def test_nested_nonretryable_sse_error_preserves_redacted_detail(
        tmp_path, monkeypatch, error_type, error_code):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-private-anthropic-key')
    monkeypatch.setenv('GOOGLE_CLOUD_API_KEY', 'fake-private-google-key')
    nested = {'type': error_type, 'code': error_code, 'param': 'model',
              'message': 'Rejected fake-private-openai-key fake-private-anthropic-key fake-private-google-key',
              'headers': {'Authorization': 'PRIVATE NEVER SAVE'}}
    # A broad outer transient code must not override specific inner failures.
    event = {'type': 'error', 'code': 'server_error', 'error': nested}
    p = provider(tmp_path, monkeypatch,
                 lambda _: httpx.Response(200, text='data: ' + json.dumps(event) + '\n\n'), max_calls=3)
    with pytest.raises(ProviderError) as exc:
        p.generate('x', {}, Review, 'r')
    assert p.calls == 1
    saved = (tmp_path / 'usage.json').read_text()
    assert error_type in saved and error_type in str(exc.value)
    assert 'fake-private-' not in saved and 'fake-private-' not in str(exc.value)
    assert 'PRIVATE NEVER SAVE' not in saved
    assert p.usage[0]['api_error']['error']['param'] == 'model'
    assert '[REDACTED]' in p.usage[0]['api_error']['error']['message']


def test_http_429_quota_error_is_not_retried(tmp_path, monkeypatch):
    p = provider(tmp_path, monkeypatch, lambda _: httpx.Response(429, json={
        'error': {'type': 'insufficient_quota', 'code': 'insufficient_quota',
                  'message': 'Quota exhausted'}}), max_calls=3)
    with pytest.raises(ProviderError, match='insufficient_quota'):
        p.generate('x', {}, Review, 'r')
    assert p.calls == 1


def test_wrapped_unknown_sse_error_retains_safe_status_fields(tmp_path, monkeypatch):
    event = {'type': 'error', 'error': {'status': 'failed', 'status_code': 403,
             'error': {'type': 'new_provider_error', 'code': 'account_policy',
                       'message': 'Request cannot proceed', 'param': None},
             'reasoning': 'PRIVATE NEVER SAVE'}}
    p = provider(tmp_path, monkeypatch,
                 lambda _: httpx.Response(200, text='data: ' + json.dumps(event) + '\n\n'), max_calls=3)
    with pytest.raises(ProviderError, match='account_policy'):
        p.generate('x', {}, Review, 'r')
    assert p.calls == 1
    nested = p.usage[0]['api_error']['error']
    assert nested['status'] == 'failed' and nested['status_code'] == 403
    assert nested['error']['type'] == 'new_provider_error'
    assert 'PRIVATE NEVER SAVE' not in (tmp_path / 'usage.json').read_text()


def test_retry_cannot_exceed_shared_cap(tmp_path, monkeypatch):
    monkeypatch.setattr('toeic_engine.openai_provider.time.sleep', lambda _: None)
    p = provider(tmp_path, monkeypatch, lambda _: httpx.Response(503, text='temporary'), max_calls=1)
    with pytest.raises(ProviderError, match='limit'):
        p.generate('x', {}, Review, 'r')
    assert p.calls == 1


def test_client_authentication_uses_only_environment(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-private-openai-key')
    p = OpenAIProvider(tmp_path)
    assert p.client.headers['Authorization'] == 'Bearer fake-private-openai-key'
    assert str(p.client.base_url) == 'https://api.openai.com'
    p.close()


def test_missing_key_is_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(ProviderError, match='OPENAI_API_KEY'):
        OpenAIProvider(tmp_path)
