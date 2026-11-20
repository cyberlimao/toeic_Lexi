import json
from types import SimpleNamespace

import httpx
import pytest

from toeic_engine.anthropic_provider import ANTHROPIC_MODEL, AnthropicProvider, ReviewerRouter, anthropic_schema
from toeic_engine.engine import PROMPT_NAMES, prompt
from toeic_engine.models import Review
from toeic_engine.provider import OutputError, ProviderError


def stream(raw, *, model=ANTHROPIC_MODEL, stop='end_turn', completed=True):
    events = [
        {'type':'message_start','message':{'id':'msg_test','model':model,'usage':{'input_tokens':10}}},
        {'type':'content_block_delta','delta':{'type':'thinking_delta','thinking':'PRIVATE NEVER SAVE'}},
        {'type':'content_block_delta','delta':{'type':'text_delta','text':raw}},
        {'type':'message_delta','delta':{'stop_reason':stop},'usage':{'output_tokens':20}},
    ]
    if completed: events.append({'type':'message_stop'})
    return ''.join('data: '+json.dumps(e)+'\n\n' for e in events)


RAW=json.dumps({'verdict':'pass','naturalness':5,'toeic_alignment':5,'explanation_quality':5,'findings':[],'item_reviews':[]})


def provider(tmp_path, monkeypatch, handler, max_calls=1):
    monkeypatch.setenv('ANTHROPIC_API_KEY','fake-private-key')
    return AnthropicProvider(tmp_path,max_calls=max_calls,client=httpx.Client(base_url='https://api.anthropic.com',transport=httpx.MockTransport(handler)))


def test_exact_model_xhigh_isolation_and_visible_only(tmp_path, monkeypatch):
    calls=[]
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200,text=stream(RAW))
    p=provider(tmp_path,monkeypatch,handler)
    assert p.generate('unchanged prompt',{'question':'frozen'},Review,'r',max_tokens=8192).passes()
    body=calls[0]
    assert body['model']==ANTHROPIC_MODEL
    assert body['output_config']['effort']=='xhigh'
    assert body['thinking']=={'type':'adaptive'}
    assert body['max_tokens']==65536 and body['stream'] is True
    assert body['messages']==[{'role':'user','content':'{"question": "frozen"}'}]
    assert all(k not in body for k in ['temperature','top_p','tools','fallbacks'])
    all_saved=''.join(f.read_text() for f in tmp_path.iterdir())
    assert 'PRIVATE NEVER SAVE' not in all_saved and 'fake-private-key' not in all_saved
    assert p.usage[0]['usage']=={'input_tokens':10,'output_tokens':20}
    with pytest.raises(ProviderError,match='limit'):p.generate('x',{},Review,'r2')
    assert len(calls)==1


@pytest.mark.parametrize('kwargs',[{'stop':'max_tokens'},{'completed':False}])
def test_partial_stream_never_counts_as_pass(tmp_path,monkeypatch,kwargs):
    p=provider(tmp_path,monkeypatch,lambda _:httpx.Response(200,text=stream(RAW,**kwargs)))
    with pytest.raises(OutputError):p.generate('x',{},Review,'r')


def test_unrequested_model_rejected(tmp_path,monkeypatch):
    p=provider(tmp_path,monkeypatch,lambda _:httpx.Response(200,text=stream(RAW,model='different-model')))
    with pytest.raises(ProviderError,match='Unexpected returned model'):p.generate('x',{},Review,'r')


def test_auth_error_redacted_and_not_retried(tmp_path,monkeypatch):
    p=provider(tmp_path,monkeypatch,lambda _:httpx.Response(401,text='invalid fake-private-key'),max_calls=3)
    with pytest.raises(ProviderError) as e:p.generate('x',{},Review,'r')
    assert 'fake-private-key' not in str(e.value)
    assert 'fake-private-key' not in (tmp_path/'usage.json').read_text()
    assert p.calls==1


def test_wire_schema_keeps_local_rating_bounds():
    s=anthropic_schema(Review)
    assert 'minimum' not in s['properties']['naturalness']
    assert Review.model_json_schema()['properties']['naturalness']['minimum']==1
    assert s['additionalProperties'] is False
    assert all(d['additionalProperties'] is False for d in s['$defs'].values() if d.get('type')=='object')


def test_author_only_to_gemini_and_all_six_review_roles_to_claude():
    class Fake:
        def __init__(self):self.calls=0;self.labels=[];self.closed=False
        def generate(self,system,payload,schema,label,**kwargs):self.calls+=1;self.labels.append(label);return label
        def close(self):self.closed=True
    a,r=Fake(),Fake()
    route=ReviewerRouter(a,r,prompt('author'),max_calls=7)
    for name in PROMPT_NAMES:route.generate(prompt(name),{},Review,name)
    assert a.labels==['author']
    assert set(r.labels)==set(PROMPT_NAMES)-{'author'}
    assert route.calls==7
    with pytest.raises(ProviderError,match='Combined'):route.generate(prompt('author'),{},Review,'author2')
    route.close();assert a.closed and r.closed
