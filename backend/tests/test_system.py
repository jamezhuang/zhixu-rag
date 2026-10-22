import json
import uuid
from pathlib import Path

from sqlalchemy import select, update

from backend.app.db import Session, Run, Conversation, Document, Meta, Message
from backend.app.config import settings
from backend.app.memory import demo_understand, context, tokens, summarize
from backend.app.knowledge import parse_file


def test_login_authorization_and_isolation(client, auth):
    assert client.get('/api/auth/me').status_code == 401
    assert client.post('/api/auth/login', json={'username': 'admin', 'password': 'wrong'}).status_code == 401
    assert client.get('/api/admin/users', headers=auth()).status_code == 403
    conv = client.post('/api/conversations', headers=auth()).json()['id']
    assert client.get(f'/api/conversations/{conv}', headers=auth('support')).status_code == 404
    assert client.post(f'/api/conversations/{conv}/requests', headers=auth('support'), json={'question': '读取别人的会话'}).status_code == 404


def test_seeded_knowledge_and_citations(ask):
    _, run = ask('试用期运营能直接确认促销折扣吗？不能的话找谁？')
    assert run['status'] == 'completed', run
    assert run['result']['mode'] == 'demo'
    assert run['result']['sources']
    assert all(s['effective_from'] <= '2026-10-21' for s in run['result']['sources'])
    assert '原文摘录' in run['result']['answer']
    assert 'password' not in json.dumps(run)


def test_date_boundary_and_missing_version(client):
    retriever = client.app.state.retriever
    old = retriever.search('试用期活动材料权限', '2026-06-30', codes=['HR-002'])
    new = retriever.search('试用期活动材料权限', '2026-07-01', codes=['HR-002'])
    assert old and all(s['version'] == '1.0' for s in old)
    assert new and all(s['version'] == '2.0' for s in new)
    assert not retriever.search('试用期活动材料权限', '2025-12-31', codes=['HR-002'])


def test_no_evidence_has_bounded_supplements(ask):
    _, run = ask('2025年12月31日的量子芯片审批规则是什么？')
    assert run['status'] == 'completed'
    assert not run['result']['sources']
    with Session() as db:
        saved = db.get(Run, run['id'])
        assert sum(e['stage'] == 'supplement' for e in saved.events) == 2


def test_idempotent_request_and_answer(client, auth):
    headers = auth()
    conv = client.post('/api/conversations', headers=headers).json()['id']
    request_id = uuid.uuid4().hex
    body = {'question': '公司业务是什么？', 'request_id': request_id}
    path = f'/api/conversations/{conv}/requests'
    assert client.post(path, headers=headers, json=body).status_code == 200
    assert client.post(path, headers=headers, json=body).status_code == 200
    assert client.post(path, headers=headers, json={**body, 'question': '别的问题'}).status_code == 409
    a = client.post(f'/api/runs/{request_id}/execute', headers=headers).json()
    b = client.post(f'/api/runs/{request_id}/execute', headers=headers).json()
    assert a['result'] == b['result']
    with Session() as db:
        assert len(db.scalars(select(Message).where(Message.request_id == request_id)).all()) == 2


def test_checkpoint_resume_after_fault(client, auth, ask):
    conv, run = ask('试用期运营促销审批找谁？', 'admin', fault='after_retrieve')
    assert run['status'] == 'failed'
    with Session() as db:
        stages = [e['stage'] for e in db.get(Run, run['id']).events]
    assert 'retrieve' in stages
    retried = client.post(f'/api/runs/{run["id"]}/execute', headers=auth('admin')).json()
    assert retried['status'] == 'completed', retried
    with Session() as db:
        saved = db.get(Run, run['id'])
        assert sum(e['stage'] == 'retrieve' for e in saved.events) == 1
        assert any(e['stage'] == 'resume' and e['detail']['resumed'] for e in saved.events)


def test_revision_change_restarts_retrieval(client, auth, ask):
    _, run = ask('活动折扣审批找谁？', 'admin', fault='after_retrieve')
    assert run['status'] == 'failed'
    with Session.begin() as db:
        db.execute(update(Meta).where(Meta.key == 'revision').values(value=Meta.value + 1))
    retried = client.post(f'/api/runs/{run["id"]}/execute', headers=auth('admin')).json()
    assert retried['status'] == 'completed'
    with Session() as db:
        saved = db.get(Run, run['id'])
        assert any(e['stage'] == 'resume' and not e['detail']['resumed'] for e in saved.events)


def test_drafts_overlap_and_published_immutability(client, auth):
    headers = auth('admin')
    doc = {'code': 'AUTH-001', 'title': '重叠版本测试', 'version': 'test-overlap', 'effective_from': '2026-08-01', 'content': '## 条款\n运营审批权限测试。'}
    response = client.post('/api/admin/documents', headers=headers, json=doc)
    assert response.status_code == 200, response.text
    id_ = response.json()['id']
    assert client.get(f'/api/documents/{id_}', headers=auth()).status_code == 404
    assert client.post(f'/api/admin/documents/{id_}/publish', headers=headers).status_code == 409
    new = {**doc, 'code': 'TEST-001', 'version': '1.0'}
    id_ = client.post('/api/admin/documents', headers=headers, json=new).json()['id']
    assert client.post(f'/api/admin/documents/{id_}/publish', headers=headers).status_code == 200
    assert client.put(f'/api/admin/documents/{id_}', headers=headers, json=new).status_code == 409
    assert client.post(f'/api/admin/documents/{id_}/disable', headers=headers).status_code == 200
    assert not client.app.state.retriever.search('运营审批权限测试', '2026-10-21', codes=['TEST-001'])


def test_invalid_dates_and_file_types(client, auth):
    headers = auth('admin')
    assert client.post('/api/admin/documents', headers=headers, json={'code': 'TEST-002', 'title': '日期测试', 'version': '1', 'effective_from': '2026-10-21', 'effective_to': '2026-01-01', 'content': '正文'}).status_code == 422
    assert parse_file('test.md', '# 标题\n测试'.encode()) == '# 标题\n测试'
    try:
        parse_file('bad.exe', b'abc')
        assert False
    except ValueError:
        pass


def test_subject_override_topic_switch_and_new_session(ask):
    conv, run = ask('我帮已经转正的客服同事问临时促销变更。')
    conv, run = ask('那他能直接确认吗？', conv=conv)
    with Session() as db:
        memory = db.get(Conversation, conv).memory
        assert memory['current']['subject']['name'] == '同事'
        assert memory['current']['subject']['probation'] is False
    ask('换个话题，我本人入职第一周学什么？', conv=conv)
    ask('回到促销问题，他能确认折扣吗？', conv=conv)
    with Session() as db:
        assert db.get(Conversation, conv).memory['current']['subject']['name'] == '同事'
    new_conv, _ = ask('我能确认促销折扣吗？')
    with Session() as db:
        assert db.get(Conversation, new_conv).memory['current']['subject']['name'] == '本人'


def test_summary_trigger_raw_retention_and_bounded_context(client, monkeypatch):
    s = settings()
    monkeypatch.setattr(s, 'memory_token_budget', 300)
    history = [{'id': i, 'role': 'user' if i % 2 else 'assistant', 'content': f'消息{i}：' + '这是活动审批的完整原始记录。' * 30} for i in range(1, 15)]
    original = json.dumps(history, ensure_ascii=False)
    digest, until, changed = summarize(client.app.state.models, history, '', 0)
    assert changed and until == 6 and digest
    assert original == json.dumps(history, ensure_ascii=False)
    payload = context(history, digest, '前面的活动怎么审批？')
    assert tokens(json.dumps({k: v for k, v in payload.items() if k != 'omitted_message_ids'}, ensure_ascii=False)) <= 300


def test_feedback_is_owned_and_upserted(client, auth, ask):
    _, run = ask('公司业务是什么？')
    path = f'/api/runs/{run["id"]}/feedback'
    assert client.post(path, headers=auth('support'), json={'helpful': True}).status_code == 404
    assert client.post(path, headers=auth(), json={'helpful': False, 'reason': '测试反馈'}).status_code == 200
    assert client.post(path, headers=auth(), json={'helpful': True}).status_code == 200


def test_disabled_user_token_is_revoked(client, auth):
    admin = auth('admin')
    body = {'username': 'revocation-test', 'name': '停用测试', 'password': 'testPass123', 'department': '运营部', 'position': '运营', 'active': True}
    row = client.post('/api/admin/users', headers=admin, json=body).json()
    login = client.post('/api/auth/login', json={'username': body['username'], 'password': body['password']}).json()
    assert client.put(f'/api/admin/users/{row["id"]}', headers=admin, json={**body, 'active': False}).status_code == 200
    assert client.get('/api/auth/me', headers={'Authorization': 'Bearer ' + login['token']}).status_code == 401


def test_thirty_reviewable_cases():
    cases = json.loads(Path('evaluation/cases.json').read_text(encoding='utf-8'))
    assert len(cases) == 30
    assert len({x['id'] for x in cases}) == 30
    assert all(x['turns'] and x['expected'] for x in cases)


def test_replacement_publication_is_atomic(client, auth, monkeypatch):
    headers = auth('admin')
    base = {'code': 'REPLACE-001', 'title': '替换发布测试', 'content': '## 规则\n活动需主管审核。', 'effective_from': '2026-01-01', 'version': '1'}
    old = client.post('/api/admin/documents', headers=headers, json=base).json()
    assert client.post(f'/api/admin/documents/{old["id"]}/publish', headers=headers).status_code == 200
    new = client.post('/api/admin/documents', headers=headers, json={**base, 'version': '2', 'effective_from': '2026-08-01'}).json()
    original = client.app.state.retriever.index_document
    def fail(*args):
        raise RuntimeError('simulated indexing failure')
    monkeypatch.setattr(client.app.state.retriever, 'index_document', fail)
    path = f'/api/admin/documents/{new["id"]}/publish?replace_previous=true'
    assert client.post(path, headers=headers).status_code == 503
    assert client.get(f'/api/documents/{old["id"]}', headers=headers).json()['effective_to'] is None
    monkeypatch.setattr(client.app.state.retriever, 'index_document', original)
    assert client.post(path, headers=headers).status_code == 200
    assert client.get(f'/api/documents/{old["id"]}', headers=headers).json()['effective_to'] == '2026-08-01'


def test_admin_experiments_are_isolated(client, auth):
    body = {'account': 'operator', 'question': '活动审批找谁？', 'experiment': 'basic'}
    assert client.post('/api/admin/experiments', headers=auth(), json=body).status_code == 403
    result = client.post('/api/admin/experiments', headers=auth('admin'), json=body).json()
    assert result['status'] == 'completed'
    assert not any(e['stage'] in ('hop', 'verify', 'assess') for e in result['events'])
    with Session() as db:
        assert db.get(Conversation, result['conversation_id']).channel == 'experiment'
