import json
import httpx
import pytest

from backend.app.providers import Models, parse_json_object, call_metrics


def test_structured_calls_disable_thinking_by_default(monkeypatch):
    model = Models()
    monkeypatch.setattr(model.s, 'deepseek_api_key', 'test-not-a-real-key')
    monkeypatch.setattr(model.s, 'deepseek_thinking', False)
    def fixture(request):
        payload = json.loads(request.content)
        assert payload['thinking'] == {'type': 'disabled'}
        assert payload['response_format'] == {'type': 'json_object'}
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '{"ok":true}'}}]})
    model.client.close()
    model.client = httpx.Client(transport=httpx.MockTransport(fixture))
    try:
        assert model.json('Return JSON', {}) == {'ok': True}
    finally:
        model.close()


@pytest.mark.parametrize('reason,content,expected', [('length', '', 'DEEPSEEK_MAX_TOKENS'), ('stop', '', 'JSON'), ('stop', 'invalid JSON', '结构化')])
def test_empty_or_truncated_model_output_is_actionable(monkeypatch, reason, content, expected):
    model = Models()
    monkeypatch.setattr(model.s, 'deepseek_api_key', 'test-not-a-real-key')
    model.client.close()
    model.client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={
        'choices': [{'finish_reason': reason, 'message': {'content': content, 'reasoning_content': 'not a final answer'}}]})))
    try:
        with pytest.raises(ValueError, match=expected):
            model.json('Return JSON', {})
    finally:
        model.close()


@pytest.mark.parametrize('content', ['\ufeff{"answer":"ok"}', '```json\n{"answer":"ok"}\n```',
    '{"answer":"first\nsecond"}', '{"answer":"first\tsecond"}'])
def test_safe_json_wrappers_and_literal_whitespace_preserve_values(content):
    result = parse_json_object(content)
    assert result['answer'] in ('ok', 'first\nsecond', 'first\tsecond')


@pytest.mark.parametrize('content', ['{"answer":"unfinished', 'prose {"answer":"ok"}',
    '{"answer":"x\x00y"}', '[]', '{"answer":"ok"} {"answer":"other"}'])
def test_malformed_or_ambiguous_json_is_not_invented(content):
    with pytest.raises(ValueError):
        parse_json_object(content)


def test_bad_format_regenerates_only_model_call_and_records_usage(monkeypatch):
    model = Models()
    monkeypatch.setattr(model.s, 'deepseek_api_key', 'test-key')
    calls, metrics = [], []
    def respond(request):
        calls.append(json.loads(request.content))
        content = '{"answer":' if len(calls) == 1 else '{"answer":"valid"}'
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':content}}]})
    model.client.close()
    model.client = httpx.Client(transport=httpx.MockTransport(respond))
    token = call_metrics.set(metrics)
    try:
        assert model.json('Answer JSON',{'question':'same question'}) == {'answer':'valid'}
        assert len(calls) == 2
        assert calls[0]['messages'][1] == calls[1]['messages'][1]
        assert [m['output_format'] for m in metrics] == ['invalid','ok']
    finally:
        call_metrics.reset(token)
        model.close()


def test_json_retry_does_not_repeat_retrieval(monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver
    from backend.app.workflow import Workflow
    model = Models()
    monkeypatch.setattr(model.s, 'app_mode', 'live')
    monkeypatch.setattr(model.s, 'deepseek_api_key', 'test-key')
    generation_calls = []
    def respond(request):
        prompt = json.loads(request.content)['messages'][0]['content']
        if prompt.startswith('理解企业问答'):
            data = {'query':'审批权限','route':'simple','memory':{},'as_of':'2026-10-22'}
        elif prompt.startswith('检查证据'):
            data = {'checks':{k:True for k in ('complete','supported','conditions','version')},'missing':[]}
        elif prompt.startswith('你是企业知识助手'):
            generation_calls.append(1)
            data = {'answer':'需主管审批。[1]'}
            if len(generation_calls)==1:
                return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'{"answer":'}}]})
        else:
            data = {'answer_supported':True,'issues':[]}
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(data)}}]})
    class Retriever:
        calls = 0
        def search(self,*a,**k):
            self.calls += 1
            return [{'id':'c1','text':'需主管审批。','refs':[],'code':'AUTH-001'}]
        def expand(self,evidence):
            return evidence
    retriever = Retriever()
    model.client.close()
    model.client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        graph = Workflow(model,retriever,InMemorySaver()).graph
        result = graph.invoke({'question':'找谁审批','profile':{},'context':{},'memory':{},'message_id':1,
            'retrieval_calls':0,'hops':0,'supplements':0,'repairs':0,'events':[]},
            {'configurable':{'thread_id':'format-retry'}})
        assert result['answer'] == '需主管审批。[1]'
        assert len(generation_calls) == 2
        assert retriever.calls == 1
    finally:
        model.close()


@pytest.mark.parametrize('mode,expected_calls', [('invalid',2),('length',1),('unauthorized',1)])
def test_format_retry_is_bounded_and_does_not_retry_auth_or_truncation(monkeypatch,mode,expected_calls):
    model = Models()
    monkeypatch.setattr(model.s,'deepseek_api_key','test-key')
    calls = []
    def respond(request):
        calls.append(1)
        if mode == 'unauthorized':
            return httpx.Response(401,json={'error':'denied'})
        return httpx.Response(200,json={'choices':[{'finish_reason':'length' if mode=='length' else 'stop',
            'message':{'content':'invalid JSON'}}]})
    model.client.close()
    model.client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        with pytest.raises((ValueError,httpx.HTTPStatusError)):
            model.json('Return JSON',{})
        assert len(calls) == expected_calls
    finally:
        model.close()
