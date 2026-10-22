import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from backend.app.config import settings
from backend.app.workflow import Workflow


class FakeModels:
    def __init__(self, fail_understand=0):
        self.failures = fail_understand
        self.understand_calls = 0

    def json(self, prompt, data):
        if '理解企业问答' in prompt:
            self.understand_calls += 1
            if self.understand_calls <= self.failures:
                raise httpx.ReadTimeout('simulated transient failure')
            return {'query': '活动审批', 'route': 'multi', 'as_of': '2026-10-21', 'memory': {'current': {'subject': {'name': '本人'}}}}
        if '检查证据是否覆盖' in prompt:
            return {'checks': {'complete': True, 'supported': True, 'conditions': True, 'version': True}, 'missing': []}
        if '你是企业知识助手' in prompt:
            return {'answer': '必须主管审核。[1]'}
        return {'answer_supported': True, 'issues': []}


class FakeRetriever:
    def search(self, query, date, codes=None):
        if codes:
            return [{'id': 'b', 'code': 'FLOW-006', 'refs': [], 'text': '主管审核', 'section': '审批', 'title': '审批流程', 'document_id': 'doc-b'}]
        return [{'id': 'a', 'code': 'AUTH-001', 'refs': ['FLOW-006'], 'text': '详见 FLOW-006', 'section': '权限', 'title': '权限', 'document_id': 'doc-a'}]

    def expand(self, evidence):
        return evidence


def state():
    return {'request_id': 'fake', 'question': '谁审批', 'profile': {}, 'context': {}, 'memory': {}, 'message_id': 1, 'events': [], 'supplements': 0, 'retrieval_calls': 0, 'hops': 0, 'repairs': 0, 'fault': '', 'revision': 0}


def test_live_branch_two_hops_and_technical_retry(monkeypatch):
    monkeypatch.setattr(settings(), 'app_mode', 'live')
    models = FakeModels(fail_understand=2)
    workflow = Workflow(models, FakeRetriever(), InMemorySaver())
    output = workflow.graph.invoke(state(), {'configurable': {'thread_id': 'live-test'}})
    assert models.understand_calls == 3
    assert output['hops'] == 2
    assert output['retrieval_calls'] == 2
    assert {x['code'] for x in output['evidence']} == {'AUTH-001', 'FLOW-006'}
    assert output['checks']['answer_supported']


def test_nontransient_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(settings(), 'app_mode', 'live')
    models = FakeModels()
    def bad(*args):
        models.understand_calls += 1
        raise ValueError('invalid output')
    models.json = bad
    workflow = Workflow(models, FakeRetriever(), InMemorySaver())
    try:
        workflow.graph.invoke(state(), {'configurable': {'thread_id': 'bad-test'}})
        assert False
    except ValueError:
        assert models.understand_calls == 1


class RewriteModels(FakeModels):
    def __init__(self, duplicate=False):
        super().__init__()
        self.rewrites = []
        self.duplicate = duplicate

    def json(self, prompt, data):
        if '理解企业问答' in prompt:
            return {'query': '试用期运营折扣审批', 'route': 'simple', 'as_of': '2026-10-21',
                    'memory': {'current': {'subject': {'position': '运营', 'probation': True}}}}
        if '检查证据是否覆盖' in prompt:
            found = any(x['id'] == 'recovered' for x in data['evidence'])
            # Deliberately omit missing: failed checks must still trigger rewrite.
            return {'checks': {k: found for k in ('complete', 'supported', 'conditions', 'version')}, 'missing': []}
        if '改写企业知识库检索查询' in prompt:
            self.rewrites.append(data)
            return {'query': data['original_query'] if self.duplicate else f'试用期运营促销折扣主管审核导师复核 条款{data["retry_number"]}'}
        return super().json(prompt, data)


class RewriteRetriever(FakeRetriever):
    def __init__(self, recover=False):
        self.queries = []
        self.recover = recover

    def search(self, query, date, codes=None):
        self.queries.append((query, date))
        if self.recover and len(self.queries) > 1:
            return [{**super().search(query, date)[0], 'id': 'recovered', 'refs': []}]
        return []


@pytest.mark.parametrize('limit,budget,expected', [(2, 6, 2), (1, 6, 1), (0, 6, 0), (3, 2, 1)])
def test_query_rewrite_stops_at_configured_limits(monkeypatch, limit, budget, expected):
    monkeypatch.setattr(settings(), 'app_mode', 'live')
    monkeypatch.setattr(settings(), 'max_query_retries', limit)
    monkeypatch.setattr(settings(), 'max_retrieval_calls', budget)
    models, retriever = RewriteModels(), RewriteRetriever()
    workflow = Workflow(models, retriever, InMemorySaver())
    result = workflow.graph.invoke(state(), {'configurable': {'thread_id': f'limit-{limit}-{budget}'}})
    assert result['supplements'] == expected
    assert len(models.rewrites) == expected
    assert len(retriever.queries) == 1 + expected
    assert result['missing']  # Exhaustion must not erase the unresolved gap.
    assert [e for e in result['events'] if e['stage'] == 'assess'][-1]['detail']['stop_reason']


def test_rewrite_recovers_evidence_and_preserves_conditions(monkeypatch):
    monkeypatch.setattr(settings(), 'app_mode', 'live')
    models, retriever = RewriteModels(), RewriteRetriever(recover=True)
    workflow = Workflow(models, retriever, InMemorySaver())
    result = workflow.graph.invoke(state(), {'configurable': {'thread_id': 'recovered'}})
    assert result['supplements'] == 1
    assert not result['missing']
    assert result['evidence'][0]['id'] == 'recovered'
    assert models.rewrites[0]['memory']['current']['subject']['probation'] is True
    assert models.rewrites[0]['previous_queries'] == ['试用期运营折扣审批']
    assert retriever.queries[1][0] != retriever.queries[0][0]
    assert all(date == '2026-10-21' for _, date in retriever.queries)


def test_duplicate_rewrite_does_not_repeat_search(monkeypatch):
    monkeypatch.setattr(settings(), 'app_mode', 'live')
    models, retriever = RewriteModels(duplicate=True), RewriteRetriever()
    result = Workflow(models, retriever, InMemorySaver()).graph.invoke(state(), {'configurable': {'thread_id': 'duplicate'}})
    assert len(retriever.queries) == 1
    assert result['supplements'] == 0
    assert next(e for e in result['events'] if e['stage'] == 'rewrite')['detail']['stop_reason']
