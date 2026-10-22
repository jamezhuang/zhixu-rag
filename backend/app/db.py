import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, String, Text, Integer, Boolean, JSON, ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import settings


def uid():
    return uuid.uuid4().hex


def now():
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'users'
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=uid)
    username: Mapped[str] = mapped_column(String(100), unique=True)
    password_hash: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(20), default='employee')
    department: Mapped[str] = mapped_column(String(100), default='运营部')
    position: Mapped[str] = mapped_column(String(100), default='店铺运营专员')
    hire_date: Mapped[str] = mapped_column(String(10), default='2026-10-01')
    probation: Mapped[bool] = mapped_column(Boolean, default=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    feishu_open_id: Mapped[str | None] = mapped_column(String(200), unique=True, nullable=True)


class Document(Base):
    __tablename__ = 'documents'
    __table_args__ = (UniqueConstraint('code', 'version'),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=uid)
    code: Mapped[str] = mapped_column(String(100), index=True)
    title: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(30))
    category: Mapped[str] = mapped_column(String(80), default='业务流程')
    effective_from: Mapped[str] = mapped_column(String(10))
    effective_to: Mapped[str | None] = mapped_column(String(10), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default='draft')
    content: Mapped[str] = mapped_column(Text)
    filename: Mapped[str] = mapped_column(String(300), default='')
    created_at: Mapped[str] = mapped_column(String(40), default=now)


class Chunk(Base):
    __tablename__ = 'chunks'
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=uid)
    document_id: Mapped[str] = mapped_column(ForeignKey('documents.id'), index=True)
    section: Mapped[str] = mapped_column(String(300))
    text: Mapped[str] = mapped_column(Text)
    location: Mapped[str] = mapped_column(String(100), default='')
    refs: Mapped[list] = mapped_column(JSON, default=list)


class Conversation(Base):
    __tablename__ = 'conversations'
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=uid)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), index=True)
    title: Mapped[str] = mapped_column(String(100), default='新的对话')
    channel: Mapped[str] = mapped_column(String(20), default='web')
    memory: Mapped[dict] = mapped_column(JSON, default=dict)
    summary: Mapped[str] = mapped_column(Text, default='')
    summary_until: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String(40), default=now)
    active_request: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_until: Mapped[float] = mapped_column(default=0.0)


class Message(Base):
    __tablename__ = 'messages'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey('conversations.id'), index=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[str] = mapped_column(String(40), default=now)
    __table_args__ = (UniqueConstraint('request_id', 'role'),)


class Run(Base):
    __tablename__ = 'runs'
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey('conversations.id'), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), index=True)
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default='pending')
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default='')
    events: Mapped[list] = mapped_column(JSON, default=list)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    trace_url: Mapped[str] = mapped_column(Text, default='')
    elapsed_ms: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[str] = mapped_column(String(40), default=now)


class Feedback(Base):
    __tablename__ = 'feedback'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey('runs.id'), unique=True)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'))
    helpful: Mapped[bool] = mapped_column(Boolean)
    reason: Mapped[str] = mapped_column(Text, default='')
    created_at: Mapped[str] = mapped_column(String(40), default=now)


class Meta(Base):
    __tablename__ = 'metadata'
    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[int] = mapped_column(Integer, default=0)


s = settings()
engine = create_engine(s.database_url, connect_args={'check_same_thread': False, 'timeout': 30} if s.database_url.startswith('sqlite') else {}, pool_pre_ping=True)
Session = sessionmaker(engine, expire_on_commit=False)


def init_db():
    Base.metadata.create_all(engine)
    with Session.begin() as db:
        if not db.get(Meta, 'revision'):
            db.add(Meta(key='revision', value=0))


def revision(db):
    return db.get(Meta, 'revision').value


def public_user(u):
    return {k: getattr(u, k) for k in ('id', 'username', 'name', 'role', 'department', 'position', 'hire_date', 'probation', 'active', 'feishu_open_id')}
