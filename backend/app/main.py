import os
import re
import uuid
from contextlib import asynccontextmanager, ExitStack
from pathlib import Path
from datetime import date

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from sqlalchemy import select, update, func
from sqlalchemy.exc import IntegrityError

from .auth import current_user, admin, verify_password, hash_password, token
from .config import settings
from .db import Session, User, Document, Chunk, Conversation, Message, Run, Feedback, Meta, init_db, public_user, revision, uid
from .knowledge import parse_file, add_chunks, serialize_document, validate_dates, publish
from .providers import Models
from .retrieval import Retriever
from .seed import seed
from .service import ChatService, own_conversation, serialize_run, safe_error
from .workflow import Workflow
from .reranker import LocalReranker


@asynccontextmanager
async def lifespan(app):
    s = settings()
    for key, value in [('LANGSMITH_TRACING', str(s.langsmith_tracing).lower()), ('LANGSMITH_API_KEY', s.langsmith_api_key), ('LANGSMITH_PROJECT', s.langsmith_project)]:
        os.environ[key] = value
    init_db()
    seed()
    with ExitStack() as stack:
        if s.database_url.startswith('postgresql'):
            from langgraph.checkpoint.postgres import PostgresSaver
            uri = s.database_url.replace('postgresql+psycopg://', 'postgresql://')
            saver = stack.enter_context(PostgresSaver.from_conn_string(uri))
        else:
            from langgraph.checkpoint.sqlite import SqliteSaver
            saver = stack.enter_context(SqliteSaver.from_conn_string(str(Path(s.data_dir) / 'checkpoints.db')))
        saver.setup()
        models = Models()
        stack.callback(models.close)
        retriever = Retriever(models)
        app.state.models, app.state.retriever = models, retriever
        app.state.chat = ChatService(Workflow(models, retriever, saver), models)
        yield


app = FastAPI(title='知序 · 新员工知识助手', version='0.1.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'], allow_methods=['*'], allow_headers=['*'])


class Login(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class EmployeeInput(BaseModel):
    username: str = Field(min_length=2, max_length=100, pattern=r'^[a-zA-Z0-9_.-]+$')
    name: str = Field(min_length=1, max_length=100)
    password: str = Field(default='', max_length=200)
    department: str = Field(default='运营部', max_length=100)
    position: str = Field(default='店铺运营专员', max_length=100)
    hire_date: date = date(2026, 10, 1)
    probation: bool = True
    active: bool = True
    feishu_open_id: str | None = Field(default=None, max_length=200)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex, pattern=r'^[a-zA-Z0-9-]{16,64}$')
    fault: str = Field(default='', pattern=r'^(|after_retrieve)$')


class FeedbackInput(BaseModel):
    helpful: bool
    reason: str = Field(default='', max_length=1000)


class DocumentInput(BaseModel):
    code: str = Field(pattern=r'^[A-Z]+-\d{3}$')
    title: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=30)
    category: str = Field(default='业务流程', max_length=80)
    effective_from: date
    effective_to: date | None = None
    content: str = Field(min_length=1, max_length=500000)


@app.get('/api/health')
def health():
    return {'status': 'ok', 'mode': settings().app_mode, 'version': '0.1.0'}


@app.post('/api/auth/login')
def login(body: Login):
    with Session() as db:
        u = db.scalar(select(User).where(User.username == body.username))
        if not u or not u.active or not verify_password(body.password, u.password_hash):
            raise HTTPException(401, '账号或密码不正确')
        return {'token': token(u), 'user': public_user(u)}


@app.get('/api/auth/me')
def me(user=Depends(current_user)):
    return public_user(user)


@app.get('/api/conversations')
def conversations(user=Depends(current_user)):
    with Session() as db:
        rows = db.scalars(select(Conversation).where(Conversation.user_id == user.id, Conversation.channel == 'web').order_by(Conversation.created_at.desc())).all()
        return [{k: getattr(x, k) for k in ('id', 'title', 'created_at')} for x in rows]


@app.post('/api/conversations')
def new_conversation(user=Depends(current_user)):
    with Session.begin() as db:
        row = Conversation(user_id=user.id)
        db.add(row)
        db.flush()
        return {'id': row.id, 'title': row.title}


@app.get('/api/conversations/{id_}')
def conversation(id_: str, user=Depends(current_user)):
    with Session() as db:
        conv = own_conversation(db, id_, user)
        messages = db.scalars(select(Message).where(Message.conversation_id == id_).order_by(Message.id)).all()
        runs = db.scalars(select(Run).where(Run.conversation_id == id_).order_by(Run.created_at)).all()
        return {'id': conv.id, 'title': conv.title,
                'messages': [{k: getattr(m, k) for k in ('id', 'request_id', 'role', 'content', 'sources', 'created_at')} for m in messages],
                'runs': [serialize_run(r) for r in runs]}


@app.post('/api/conversations/{id_}/requests')
def create_request(id_: str, body: Question, request: Request, user=Depends(current_user)):
    if not body.question.strip():
        raise HTTPException(422, '问题不能为空')
    return request.app.state.chat.create(user, id_, body.question.strip(), body.request_id, body.fault)


@app.post('/api/runs/{id_}/execute')
def execute_request(id_: str, request: Request, user=Depends(current_user)):
    return request.app.state.chat.execute(user, id_)


@app.post('/api/runs/{id_}/cancel')
def cancel_request(id_: str, user=Depends(current_user)):
    with Session.begin() as db:
        run = db.get(Run, id_, with_for_update=True)
        if not run or run.user_id != user.id:
            raise HTTPException(404, '请求不存在')
        if run.status != 'completed':
            run.status = 'cancelled'
        return serialize_run(run)


@app.post('/api/runs/{id_}/feedback')
def feedback(id_: str, body: FeedbackInput, user=Depends(current_user)):
    with Session.begin() as db:
        run = db.get(Run, id_)
        if not run or run.user_id != user.id or run.status != 'completed':
            raise HTTPException(404, '已完成的问答不存在')
        row = db.scalar(select(Feedback).where(Feedback.run_id == id_))
        if not row:
            row = Feedback(run_id=id_, user_id=user.id, helpful=body.helpful)
            db.add(row)
        row.helpful, row.reason = body.helpful, body.reason
        return {'ok': True}


@app.get('/api/documents')
def documents(user=Depends(current_user)):
    with Session() as db:
        query = select(Document).order_by(Document.code, Document.effective_from.desc())
        if user.role != 'admin':
            query = query.where(Document.status == 'published')
        rows = db.scalars(query).all()
        return [serialize_document(x) for x in rows]


@app.get('/api/documents/{id_}')
def document(id_: str, user=Depends(current_user)):
    with Session() as db:
        doc = db.get(Document, id_)
        if not doc:
            raise HTTPException(404, '文档不存在')
        if user.role != 'admin' and doc.status != 'published':
            # Historical citations are authorized by ownership of an actual answer.
            owned = db.execute(select(Message).join(Conversation, Message.conversation_id == Conversation.id)
                               .where(Conversation.user_id == user.id, Message.role == 'assistant')).scalars().all()
            if not any(any(x.get('document_id') == id_ for x in m.sources) for m in owned):
                raise HTTPException(404, '文档不可访问')
        result = serialize_document(doc, True)
        result['chunks'] = [{'id': c.id, 'section': c.section, 'text': c.text, 'location': c.location, 'refs': c.refs}
                            for c in db.scalars(select(Chunk).where(Chunk.document_id == id_)).all()]
        return result


def create_doc(body, filename=''):
    start, end = body.effective_from.isoformat(), body.effective_to.isoformat() if body.effective_to else None
    validate_dates(start, end)
    try:
        with Session.begin() as db:
            doc = Document(**body.model_dump(exclude={'effective_from', 'effective_to'}), effective_from=start, effective_to=end, filename=filename)
            db.add(doc)
            db.flush()
            if not add_chunks(db, doc):
                raise ValueError('文档中没有可检索正文')
            return serialize_document(doc, True)
    except IntegrityError:
        raise ValueError('文档编号与版本组合已存在')


@app.post('/api/admin/documents')
def add_document(body: DocumentInput, user=Depends(admin)):
    try:
        return create_doc(body)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post('/api/admin/documents/upload')
def upload(file: UploadFile = File(...), code: str = Form(...), title: str = Form(...), version: str = Form(...),
           effective_from: str = Form(...), effective_to: str = Form(''), category: str = Form('业务流程'), user=Depends(admin)):
    data = file.file.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, '文件不能超过 10 MB')
    try:
        content = parse_file(file.filename or '', data)
        body = DocumentInput(code=code, title=title, version=version, effective_from=effective_from,
                             effective_to=effective_to or None, category=category, content=content)
        result = create_doc(body, Path(file.filename or 'document').name)
        # Use internal UUID; never trust a user-supplied filesystem path.
        dest = Path(settings().data_dir) / 'uploads'
        dest.mkdir(exist_ok=True)
        (dest / f'{result["id"]}.{(file.filename or "txt").rsplit(".", 1)[-1].lower()}').write_bytes(data)
        return result
    except Exception as exc:
        raise HTTPException(422, '文档解析或保存失败：' + safe_error(exc))


@app.put('/api/admin/documents/{id_}')
def edit_document(id_: str, body: DocumentInput, user=Depends(admin)):
    from sqlalchemy import delete
    try:
        with Session.begin() as db:
            d = db.get(Document, id_)
            if not d:
                raise HTTPException(404, '文档不存在')
            # Keep every indexed or cited version immutable. Create a new one.
            if d.status != 'draft':
                raise HTTPException(409, '已发布或停用的版本不可修改，请新建版本')
            start, end = body.effective_from.isoformat(), body.effective_to.isoformat() if body.effective_to else None
            validate_dates(start, end)
            for key, value in body.model_dump(exclude={'effective_from', 'effective_to'}).items():
                setattr(d, key, value)
            d.effective_from, d.effective_to = start, end
            db.execute(delete(Chunk).where(Chunk.document_id == id_))
            add_chunks(db, d)
            return serialize_document(d, True)
    except (ValueError, IntegrityError) as exc:
        raise HTTPException(422, '有效期或文档编号版本无效')


class DocumentCategoryInput(BaseModel):
    model_config = {'extra': 'forbid'}
    category: Literal['业务流程', '公司业务', '岗位职责', '入职制度', '经营知识']


@app.patch('/api/admin/documents/{id_}/category')
def update_document_category(id_: str, body: DocumentCategoryInput, user=Depends(admin)):
    # Category is catalog metadata, independent of immutable policy content and
    # retrieval indexes. Do not mutate chunks, dates, publication or revision.
    with Session.begin() as db:
        doc = db.get(Document, id_)
        if not doc:
            raise HTTPException(404, '文档不存在')
        doc.category = body.category
        return serialize_document(doc)


@app.post('/api/admin/documents/{id_}/publish')
def publish_document(id_: str, request: Request, replace_previous: bool = False, user=Depends(admin)):
    try:
        with Session.begin() as db:
            d = db.get(Document, id_)
            if not d:
                raise HTTPException(404, '文档不存在')
            publish(db, d, request.app.state.retriever, replace_previous=replace_previous)
            return serialize_document(d)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, '索引未完成，文档尚未发布。' + safe_error(exc))


@app.post('/api/admin/documents/{id_}/disable')
def disable_document(id_: str, user=Depends(admin)):
    with Session.begin() as db:
        d = db.get(Document, id_)
        if not d:
            raise HTTPException(404, '文档不存在')
        d.status = 'disabled'
        db.execute(update(Meta).where(Meta.key == 'revision').values(value=Meta.value + 1))
        return serialize_document(d)


@app.get('/api/admin/users')
def users(user=Depends(admin)):
    with Session() as db:
        return [public_user(u) for u in db.scalars(select(User).order_by(User.username)).all()]


def save_employee(body, id_=None):
    try:
        with Session.begin() as db:
            row = db.get(User, id_) if id_ else None
            if id_ and not row:
                raise HTTPException(404, '员工不存在')
            if row and row.role == 'admin':
                raise HTTPException(409, '此处只允许编辑员工账号')
            if not row:
                if len(body.password) < 8:
                    raise HTTPException(422, '新账号密码至少 8 位')
                row = User(role='employee', password_hash=hash_password(body.password))
                db.add(row)
            for key, value in body.model_dump(exclude={'password', 'hire_date'}).items():
                setattr(row, key, value or None if key == 'feishu_open_id' else value)
            row.hire_date = body.hire_date.isoformat()
            if body.password:
                if len(body.password) < 8:
                    raise HTTPException(422, '密码至少 8 位')
                row.password_hash = hash_password(body.password)
            db.flush()
            return public_user(row)
    except IntegrityError:
        raise HTTPException(409, '账号或飞书身份已被使用')


@app.post('/api/admin/users')
def add_user(body: EmployeeInput, user=Depends(admin)):
    return save_employee(body)


@app.put('/api/admin/users/{id_}')
def edit_user(id_: str, body: EmployeeInput, user=Depends(admin)):
    return save_employee(body, id_)


@app.get('/api/admin/runs')
def runs(user=Depends(admin)):
    with Session() as db:
        rows = db.scalars(select(Run).order_by(Run.created_at.desc()).limit(200)).all()
        result = []
        for r in rows:
            item = serialize_run(r, True)
            item['user_name'] = db.get(User, r.user_id).name
            fb = db.scalar(select(Feedback).where(Feedback.run_id == r.id))
            item['feedback'] = {'helpful': fb.helpful, 'reason': fb.reason} if fb else None
            result.append(item)
        return result


@app.get('/api/admin/overview')
def overview(user=Depends(admin)):
    with Session() as db:
        count = lambda cls: db.scalar(select(func.count()).select_from(cls))
        runs = db.scalars(select(Run)).all()
        return {'documents': count(Document), 'employees': count(User), 'questions': len(runs),
                'published': db.scalar(select(func.count()).select_from(Document).where(Document.status == 'published')),
                'completed': sum(r.status == 'completed' for r in runs), 'failed': sum(r.status == 'failed' for r in runs),
                'feedback': count(Feedback), 'revision': revision(db), 'mode': settings().app_mode,
                'models': {'generation': settings().deepseek_model, 'embedding': settings().embedding_model, 'rerank': settings().rerank_model},
                'configured': {'generation': bool(settings().deepseek_api_key), 'embedding': bool(settings().siliconflow_api_key),
                               'rerank_local': LocalReranker().ready(),
                               'langsmith': settings().langsmith_tracing and bool(settings().langsmith_api_key),
                               'feishu': bool(settings().feishu_app_id and settings().feishu_app_secret)}}


class ExperimentInput(BaseModel):
    account: str
    question: str = Field(min_length=1, max_length=4000)
    experiment: Literal['full', 'basic', 'recent_only', 'no_reflection'] = 'full'
    conversation_id: str | None = None


@app.post('/api/admin/experiments')
def experiment(body: ExperimentInput, request: Request, user=Depends(admin)):
    with Session.begin() as db:
        target = db.scalar(select(User).where(User.username == body.account, User.active.is_(True)))
        if not target:
            raise HTTPException(404, '实验员工不存在')
        conv = db.get(Conversation, body.conversation_id) if body.conversation_id else None
        if body.conversation_id and (not conv or conv.user_id != target.id or conv.channel != 'experiment'):
            raise HTTPException(404, '实验会话不存在')
        if conv is None:
            conv = Conversation(user_id=target.id, channel='experiment')
            db.add(conv)
            db.flush()
        conv_id = conv.id
    run = request.app.state.chat.create(target, conv_id, body.question, uid(), experiment=body.experiment)
    result = request.app.state.chat.execute(target, run['id'])
    with Session() as db:
        return serialize_run(db.get(Run, run['id']), True)
