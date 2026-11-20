import json
from types import SimpleNamespace

import pytest
from google.genai import errors, types

from toeic_engine.models import Review, Unit
from toeic_engine.provider import MODEL, OutputError, ProviderError, VertexProvider, generation_schema, safe_error


def test_wire_schema_preserves_property_names_and_keeps_local_constraints():
    compact = generation_schema(Unit)
    assert 'title' in compact['properties']
    assert all(x in compact['properties'] for x in compact['required'])
    assert 'maxItems' not in compact['properties']['questions']
    assert Unit.model_json_schema()['properties']['questions']['maxItems'] == 30
    for item in compact['$defs'].values():
        assert all(x in item['properties'] for x in item.get('required', []))


def test_error_redacts_environment_key_and_query_key(monkeypatch):
    monkeypatch.setenv('GOOGLE_CLOUD_API_KEY', 'a-secret-value')
    exc = SimpleNamespace(message='bad a-secret-value url?key=another-secret&x=1')
    assert 'a-secret-value' not in safe_error(exc)
    assert 'another-secret' not in safe_error(exc)


def test_provider_uses_vertex_high_and_budget_and_no_sampling(monkeypatch, tmp_path):
    monkeypatch.setenv('GOOGLE_CLOUD_API_KEY', 'test-only-key')
    captured = {}
    raw = json.dumps({'verdict': 'pass', 'naturalness': 5, 'toeic_alignment': 5, 'explanation_quality': 5, 'findings': [], 'item_reviews': []})
    response = types.GenerateContentResponse(model_version=MODEL, candidates=[types.Candidate(finish_reason='STOP', content=types.Content(parts=[types.Part(text=raw)]))])
    def generate(**kwargs):
        captured.update(kwargs)
        return response
    def client(**kwargs):
        captured['client'] = kwargs
        return SimpleNamespace(models=SimpleNamespace(generate_content=generate), close=lambda: None)
    monkeypatch.setattr('toeic_engine.provider.genai.Client', client)
    provider = VertexProvider(tmp_path, max_calls=1)
    assert provider.generate('review', {}, Review, 'r').passes()
    config = captured['config'].model_dump(exclude_none=True)
    assert captured['client']['vertexai'] is True
    assert captured['model'] == MODEL
    assert config['thinking_config']['thinking_level'] == 'HIGH'
    assert all(k not in config for k in ['temperature', 'top_p', 'top_k', 'candidate_count', 'thinking_budget'])
    assert 'test-only-key' not in ''.join(p.read_text() for p in tmp_path.iterdir())
    with pytest.raises(ProviderError, match='limit'):
        provider.generate('review', {}, Review, 'r2')


def test_non_stop_response_cannot_become_review(monkeypatch, tmp_path):
    monkeypatch.setenv('GOOGLE_CLOUD_API_KEY', 'test-only-key')
    response = types.GenerateContentResponse(candidates=[types.Candidate(finish_reason='MAX_TOKENS', content=types.Content(parts=[types.Part(text='{}')]))])
    monkeypatch.setattr('toeic_engine.provider.genai.Client', lambda **kwargs: SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: response), close=lambda: None))
    with pytest.raises(OutputError, match='MAX_TOKENS'):
        VertexProvider(tmp_path).generate('x', {}, Review, 'r')
