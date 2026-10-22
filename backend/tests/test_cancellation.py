import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from backend.app.cancellation import cancel_check, check_cancelled, GenerationCancelled
from backend.app.db import Session, Run, Message, Conversation
from backend.app.main import app
from backend.app.providers import Models
from sqlalchemy import select


def test_running_request_can_stop_without_answer_or_later_nodes(client, auth, monkeypatch):
    headers = auth()
    conv = client.post('/api/conversations', headers=headers).json()['id']
    run = client.post(f'/api/conversations/{conv}/requests', headers=headers,
                      json={'question': '测试停止生成'}).json()
    entered = threading.Event()
    # Pausing the retriever exercises a real graph node and the execution transaction.
    retriever = app.state.chat.workflow.retriever
    def blocking_search(*args, **kwargs):
        entered.set()
        for _ in range(200):
            check_cancelled()
            threading.Event().wait(0.01)
        raise AssertionError('Cancellation was not observed')
    monkeypatch.setattr(retriever, 'search', blocking_search)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(client.post, f'/api/runs/{run["id"]}/execute', headers=headers)
        assert entered.wait(5)
        # Another employee cannot cancel this request.
        assert client.post(f'/api/runs/{run["id"]}/cancel', headers=auth('support')).status_code == 404
        assert client.post(f'/api/runs/{run["id"]}/cancel', headers=headers).json()['status'] == 'cancelled'
        assert future.result(timeout=5).json()['status'] == 'cancelled'
    with Session() as db:
        saved = db.get(Run, run['id'])
        assert saved.status == 'cancelled'
        assert saved.events[-1]['stage'] == 'cancelled'
        assert not db.scalar(select(Message.id).where(Message.request_id == run['id'], Message.role == 'assistant'))
        assert db.get(Conversation, conv).active_request is None
    assert client.post(f'/api/runs/{run["id"]}/execute', headers=headers).json()['status'] == 'cancelled'
    assert client.post(f'/api/conversations/{conv}/requests', headers=headers,
                       json={'question': '停止后继续提问'}).status_code == 200


def test_cancel_closes_inflight_model_request(monkeypatch):
    started, closed = threading.Event(), threading.Event()
    async def blocked(request):
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            closed.set()
    factory = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: factory(transport=httpx.MockTransport(blocked), **kwargs))
    def checker():
        if started.is_set():
            raise GenerationCancelled()
    token = cancel_check.set(checker)
    model = Models()
    try:
        with pytest.raises(GenerationCancelled):
            asyncio.run(model._cancellable_post('https://test.invalid', {}, {}, 60))
        assert closed.is_set()
    finally:
        cancel_check.reset(token)
        model.close()


def test_completed_answer_is_not_cancelled(client, auth, ask):
    _, run = ask('公司提供什么服务？')
    assert run['status'] == 'completed'
    result = client.post(f'/api/runs/{run["id"]}/cancel', headers=auth()).json()
    assert result['status'] == 'completed'
