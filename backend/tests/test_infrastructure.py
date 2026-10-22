"""Opt-in real infrastructure tests. No paid API or real Feishu calls."""
import hashlib
import os
import uuid
from typing import TypedDict

import pytest

pytestmark = pytest.mark.skipif(os.getenv('RUN_INFRA_TESTS') != 'true', reason='Set RUN_INFRA_TESTS=true with local PostgreSQL and Milvus running')


def test_postgres_checkpoints_survive_new_connection():
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.graph import StateGraph, START, END

    uri = os.getenv('TEST_POSTGRES_URI', 'postgresql://zhixu:zhixu_local_only@127.0.0.1:55432/zhixu')
    schema = 'zhixu_verify_' + uuid.uuid4().hex[:12]
    class State(TypedDict):
        value: int

    def build(saver, fail):
        graph = StateGraph(State)
        graph.add_node('prepare', lambda s: {'value': s['value'] + 1})
        def finish(s):
            if fail:
                raise RuntimeError('simulated process boundary')
            return {'value': s['value'] + 10}
        graph.add_node('finish', finish)
        graph.add_edge(START, 'prepare')
        graph.add_edge('prepare', 'finish')
        graph.add_edge('finish', END)
        return graph.compile(checkpointer=saver)

    config = {'configurable': {'thread_id': 'persistent-test'}}
    with psycopg.connect(uri, autocommit=True, row_factory=dict_row, connect_timeout=5) as conn:
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        conn.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
        saver = PostgresSaver(conn)
        saver.setup()
        graph = build(saver, True)
        with pytest.raises(RuntimeError):
            graph.invoke({'value': 0}, config)
        assert graph.get_state(config).values['value'] == 1
    # A fresh connection and fresh graph must use persisted values and pending node.
    with psycopg.connect(uri, autocommit=True, row_factory=dict_row, connect_timeout=5) as conn:
        conn.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
        graph = build(PostgresSaver(conn), False)
        assert graph.invoke(None, config)['value'] == 11
        # Only the UUID schema created in this test is removed.
        conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))


def test_real_milvus_index_filter_and_search(client, monkeypatch):
    from sqlalchemy import select
    from backend.app.config import settings
    from backend.app.db import Session, Document, Chunk
    from backend.app.retrieval import Retriever

    s = settings()
    monkeypatch.setattr(s, 'app_mode', 'live')
    monkeypatch.setattr(s, 'milvus_collection', 'zhixu_verify_' + uuid.uuid4().hex[:12])
    class FixtureModels:
        def embed(self, texts):
            # Deterministic nonsemantic fixtures: this verifies plumbing only.
            vectors = []
            for text in texts:
                digest = hashlib.sha256(text.encode()).digest()
                vectors.append([(digest[i % 32] - 127.5) / 127.5 for i in range(s.embedding_dim)])
            return vectors

        def rerank(self, query, candidates):
            return candidates

    fixture = FixtureModels()
    retriever = Retriever(fixture, reranker=fixture)
    with Session() as db:
        doc = db.scalar(select(Document).where(Document.code == 'BIZ-001'))
        chunks = db.scalars(select(Chunk).where(Chunk.document_id == doc.id)).all()
        retriever.index_document(doc, chunks)
    try:
        result = retriever.search('公司为品牌提供什么服务', '2026-10-21', codes=['BIZ-001'])
        assert result and all(x['code'] == 'BIZ-001' for x in result)
        assert not retriever.search('公司服务', '2025-01-01', codes=['BIZ-001'])
        names = retriever.milvus().list_indexes(s.milvus_collection)
        info = retriever.milvus().describe_index(s.milvus_collection, names[0])
        assert info['index_type'] == 'HNSW'
    finally:
        retriever.milvus().drop_collection(s.milvus_collection)
