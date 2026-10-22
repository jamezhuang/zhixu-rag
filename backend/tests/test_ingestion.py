from types import SimpleNamespace

import httpx
import pytest

from backend.app.config import settings
from backend.app.retrieval import Retriever


class FixtureStore:
    def __init__(self):
        self.writes = []
        self.flushed = False

    def upsert(self, **kwargs):
        self.writes.append(kwargs)

    def flush(self, collection):
        self.flushed = True


def fixtures(monkeypatch, embed):
    monkeypatch.setattr(settings(), 'app_mode', 'live')
    monkeypatch.setattr('backend.app.retrieval.time.sleep', lambda _: None)
    retriever = Retriever(SimpleNamespace(embed=embed))
    store = FixtureStore()
    monkeypatch.setattr(retriever, 'milvus', lambda: store)
    doc = SimpleNamespace(id='document', title='Example policy')
    chunks = [SimpleNamespace(id='chunk', section='Scope', text='A complete policy clause.')]
    return retriever, store, doc, chunks


def test_publication_recovers_from_transient_embedding_connections(monkeypatch):
    calls = []
    def embed(texts):
        calls.append(texts)
        if len(calls) < 3:
            raise httpx.ConnectError('simulated transient TLS connection failure')
        return [[0.1, 0.2]]
    retriever, store, doc, chunks = fixtures(monkeypatch, embed)
    retriever.index_document(doc, chunks)
    assert len(calls) == 3
    assert len(store.writes) == 1 and store.flushed


def test_publication_does_not_retry_authentication_errors(monkeypatch):
    calls = []
    def embed(texts):
        calls.append(texts)
        response = httpx.Response(401, request=httpx.Request('POST', 'https://example.invalid/embeddings'))
        response.raise_for_status()
    retriever, store, doc, chunks = fixtures(monkeypatch, embed)
    with pytest.raises(httpx.HTTPStatusError):
        retriever.index_document(doc, chunks)
    assert len(calls) == 1 and not store.writes


def test_publication_stops_after_three_transient_failures(monkeypatch):
    calls = []
    def embed(texts):
        calls.append(texts)
        raise httpx.ReadTimeout('simulated persistent timeout')
    retriever, store, doc, chunks = fixtures(monkeypatch, embed)
    with pytest.raises(httpx.ReadTimeout):
        retriever.index_document(doc, chunks)
    assert len(calls) == 3 and not store.writes
