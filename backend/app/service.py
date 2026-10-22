import json
import threading
import time
import uuid
from contextlib import contextmanager

import httpx
from fastapi import HTTPException
from sqlalchemy import select, update, or_

from .config import settings
from .db import Session, Conversation, Message, Run, User, revision, now, public_user
from .memory import context, summarize
from .providers import call_metrics, call_deadline
from .compression import compression_events
from .cancellation import cancel_check, check_cancelled, GenerationCancelled


def serialize_run(run, detail=False):
    result = {k: getattr(run, k) for k in ('id', 'conversation_id', 'question', 'status', 'error', 'elapsed_ms', 'created_at')}
    result['result'] = {k: v for k, v in run.result.items() if not k.startswith('_')}
    if detail:
        result.update(events=run.events, trace_url=run.trace_url, revision=run.revision, metrics=run.metrics)
    return result


def own_conversation(db, id_, user):
    conv = db.get(Conversation, id_)
    if not conv or conv.user_id != user.id:
        raise HTTPException(404, '会话不存在')
    return conv


def safe_error(exc):
    if isinstance(exc, httpx.HTTPStatusError):
        return f'模型接口返回 HTTP {exc.response.status_code}，请检查配置、额度或服务状态后重试'
    if isinstance(exc, httpx.TimeoutException):
        return '模型服务超时，已用完本轮技术重试，可重试本次问题'
    if isinstance(exc, httpx.NetworkError):
        return '模型服务网络连接失败，请检查网络后重试'
    if 'injected interruption' in str(exc):
        return '演示故障：已在检索完成后中断。可从检查点重试。'
    if isinstance(exc, ValueError):
        return str(exc)[:250]
    return '执行未完成，检查点已保留；请重试本次问题，管理员可检查运行记录与配置'


class ChatService:
    def __init__(self, workflow, models):
        self.workflow, self.models = workflow, models

    def create(self, user, conv_id, question, request_id, fault='', experiment='full'):
        with Session.begin() as db:
            conv = own_conversation(db, conv_id, user)
            existing = db.get(Run, request_id)
            if existing:
                if existing.user_id != user.id or existing.conversation_id != conv_id or existing.question != question:
                    raise HTTPException(409, '请求编号已被使用')
                return serialize_run(existing)
            active = db.scalar(select(Run).where(Run.conversation_id == conv_id, Run.status.in_(['pending', 'running', 'failed'])).limit(1))
            if active:
                raise HTTPException(409, '请先重试或放弃上一条未完成的问题')
            if fault and not (settings().enable_fault_injection and user.role == 'admin'):
                raise HTTPException(403, '故障注入仅供启用开发开关的管理员使用')
            run = Run(id=request_id, user_id=user.id, conversation_id=conv_id, question=question,
                      revision=revision(db), result={'_fault': fault, '_experiment': experiment})
            db.add(run)
            db.add(Message(conversation_id=conv_id, request_id=request_id, role='user', content=question))
            if conv.title == '新的对话':
                conv.title = question[:28]
            db.flush()
            return serialize_run(run)

    @contextmanager
    def lease(self, run_id, conversation_id):
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(10):
                with Session.begin() as db:
                    db.execute(update(Conversation).where(Conversation.id == conversation_id, Conversation.active_request == run_id)
                               .values(lease_until=time.time() + 60))

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=2)
            with Session.begin() as db:
                db.execute(update(Conversation).where(Conversation.id == conversation_id, Conversation.active_request == run_id)
                           .values(active_request=None, lease_until=0))

    def execute(self, user, run_id):
        s = settings()
        with Session.begin() as db:
            run = db.get(Run, run_id, with_for_update=True)
            if not run or run.user_id != user.id:
                raise HTTPException(404, '请求不存在')
            if run.status in ('completed', 'cancelled'):
                return serialize_run(run)
            conv_id = run.conversation_id
            claim = db.execute(update(Conversation).where(Conversation.id == conv_id,
                or_(Conversation.active_request.is_(None), Conversation.lease_until < time.time())).values(active_request=run.id, lease_until=time.time() + 60))
            if claim.rowcount != 1:
                raise HTTPException(409, '该会话有正在执行的请求，请稍后刷新')
            run.status, run.error = 'running', ''
        with self.lease(run_id, conv_id):
            metrics = []
            compressions = []
            compression_token = compression_events.set(compressions)
            metric_token = call_metrics.set(metrics)
            deadline_token = call_deadline.set(time.monotonic() + s.request_timeout_seconds)
            def check_status():
                with Session() as db:
                    if db.scalar(select(Run.status).where(Run.id == run_id)) == 'cancelled':
                        raise GenerationCancelled('用户已停止生成')
            cancellation_token = cancel_check.set(check_status)
            started = time.perf_counter()
            config = {'configurable': {'thread_id': f'{user.id}:{run_id}'}, 'recursion_limit': 40,
                      'run_id': uuid.uuid4(), 'run_name': 'zhixu-rag', 'metadata': {'request_id': run_id, 'conversation_id': conv_id, 'mode': s.app_mode}}
            try:
                with Session() as db:
                    run, conv = db.get(Run, run_id), db.get(Conversation, conv_id)
                    current_revision = revision(db)
                    messages = db.scalars(select(Message).where(Message.conversation_id == conv_id).order_by(Message.id)).all()
                    history = [{'id': x.id, 'role': x.role, 'content': x.content} for x in messages if x.request_id != run_id]
                    message_id = next(x.id for x in messages if x.request_id == run_id and x.role == 'user')
                    experiment = run.result.get('_experiment', 'full')
                    initial = {'request_id': run_id, 'question': run.question, 'profile': public_user(user), 'experiment': experiment,
                               'context': context(history, conv.summary, run.question), 'memory': conv.memory,
                               'message_id': message_id, 'evidence': [], 'checks': {}, 'missing': [],
                               'supplements': 0, 'retrieval_calls': 0, 'hops': 0, 'answer': '', 'events': [], 'repairs': 0,
                               'fault': run.result.get('_fault', ''), 'revision': current_revision}
                    if experiment == 'recent_only':
                        initial['memory'] = {}
                        initial['context'] = {'summary': '', 'recalled_originals': [], 'recent': history[-s.recent_rounds * 2:]}
                snapshot = self.workflow.graph.get_state(config)
                can_resume = bool(snapshot.values) and snapshot.values.get('revision') == current_revision and snapshot.values.get('profile') == initial['profile']
                if can_resume and not snapshot.next:
                    result = snapshot.values  # graph completed before business transaction committed
                else:
                    result = self.workflow.graph.invoke(None if can_resume else initial, config=config)
                with Session() as db:
                    if revision(db) != current_revision:
                        raise ValueError('问答期间知识库版本发生变化，请重试以重新检索')
                with Session.begin() as db:
                    run = db.get(Run, run_id, with_for_update=True)
                    if run.status == 'cancelled':
                        raise GenerationCancelled('用户已停止生成')
                    conv = db.get(Conversation, conv_id)
                    # No password or provider configuration enters final responses.
                    sources = [{k: v for k, v in x.items() if k not in ('fusion_score', 'rerank_score')} for x in result['evidence']]
                    run.result = {**{k: v for k, v in run.result.items() if k.startswith('_')},
                                  'answer': result['answer'], 'sources': sources, 'mode': s.app_mode}
                    run.events = result['events'] + list(compressions) + [{'stage': 'resume', 'detail': {'resumed': can_resume, 'experiment': experiment}}]
                    run.metrics = run.metrics + list(metrics)
                    run.revision, run.elapsed_ms, run.status = current_revision, int((time.perf_counter() - started) * 1000), 'completed'
                    conv.memory = result['memory']
                    if not db.scalar(select(Message.id).where(Message.request_id == run_id, Message.role == 'assistant')):
                        db.add(Message(conversation_id=conv_id, request_id=run_id, role='assistant', content=result['answer'], sources=sources))
                saved_metric_count = len(metrics)
                saved_compression_count = len(compressions)
                # Summary failures must not erase a completed answer or the old digest.
                try:
                    with Session() as db:
                        conv = db.get(Conversation, conv_id)
                        messages = db.scalars(select(Message).where(Message.conversation_id == conv_id).order_by(Message.id)).all()
                        summary, until, changed = summarize(self.models, [{'id': x.id, 'role': x.role, 'content': x.content} for x in messages], conv.summary, conv.summary_until)
                    if changed:
                        with Session.begin() as db:
                            conv = db.get(Conversation, conv_id)
                            conv.summary, conv.summary_until = summary, until
                except Exception:
                    with Session.begin() as db:
                        run = db.get(Run, run_id)
                        run.events = run.events + [{'stage': 'summary', 'detail': '摘要更新失败，旧摘要与原始消息均保留'}]
                if s.langsmith_tracing and s.langsmith_api_key:
                    try:
                        from langsmith import Client
                        url = Client().get_run_url(run_id=config['run_id'])
                        with Session.begin() as db:
                            db.get(Run, run_id).trace_url = url
                    except Exception:
                        pass
                if len(metrics) > saved_metric_count:
                    with Session.begin() as db:
                        run = db.get(Run, run_id)
                        run.metrics = run.metrics + metrics[saved_metric_count:]
                if len(compressions) > saved_compression_count:
                    with Session.begin() as db:
                        run = db.get(Run, run_id)
                        run.events = run.events + compressions[saved_compression_count:]
            except Exception as exc:
                snapshot = self.workflow.graph.get_state(config)
                with Session.begin() as db:
                    run = db.get(Run, run_id, with_for_update=True)
                    cancelled = run.status == 'cancelled' or isinstance(exc, GenerationCancelled)
                    run.status, run.error = ('cancelled', '') if cancelled else ('failed', safe_error(exc))
                    run.elapsed_ms = int((time.perf_counter() - started) * 1000)
                    run.metrics = run.metrics + list(metrics)
                    run.events = snapshot.values.get('events', []) + list(compressions) + [{'stage': 'cancelled' if cancelled else 'failed', 'detail': {'error_type': type(exc).__name__, 'pending_nodes': list(snapshot.next)}}]
            with Session() as db:
                output = serialize_run(db.get(Run, run_id))
            call_metrics.reset(metric_token)
            call_deadline.reset(deadline_token)
            compression_events.reset(compression_token)
            cancel_check.reset(cancellation_token)
            return output
