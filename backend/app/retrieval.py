import re
import threading
import time

import jieba
from rank_bm25 import BM25L
from sqlalchemy import select

from .config import settings
from .db import Session, Document, Chunk
from .providers import Models, transient
from .reranker import LocalReranker

jieba.setLogLevel(30)


def tokenize(text):
    words = [x.lower() for x in jieba.lcut(text) if x.strip() and re.search(r'[\w\u4e00-\u9fff]', x)]
    words += [x.lower() for x in re.findall(r'[A-Z]+-\d{3}', text)]
    return words


def source(chunk, doc):
    return {'id': chunk.id, 'document_id': doc.id, 'code': doc.code, 'title': doc.title, 'version': doc.version,
            'effective_from': doc.effective_from, 'effective_to': doc.effective_to,
            'section': chunk.section, 'text': chunk.text, 'location': chunk.location, 'refs': chunk.refs}


class Retriever:
    def __init__(self, models: Models, reranker=None):
        self.s, self.models = settings(), models
        self._client = None
        self.lock = threading.Lock()
        self.reranker = reranker if reranker is not None else LocalReranker()

    def milvus(self):
        if self._client is None:
            from pymilvus import MilvusClient, DataType
            with self.lock:
                if self._client is not None:
                    return self._client
                client = MilvusClient(uri=self.s.milvus_uri, timeout=20)
                if not client.has_collection(self.s.milvus_collection):
                    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
                    schema.add_field('id', DataType.VARCHAR, is_primary=True, max_length=64)
                    schema.add_field('document_id', DataType.VARCHAR, max_length=64)
                    schema.add_field('vector', DataType.FLOAT_VECTOR, dim=self.s.embedding_dim)
                    schema.add_field('model', DataType.VARCHAR, max_length=150)
                    indexes = client.prepare_index_params()
                    indexes.add_index(field_name='vector', index_type='HNSW', metric_type='COSINE', params={'M': 16, 'efConstruction': 128})
                    client.create_collection(self.s.milvus_collection, schema=schema, index_params=indexes)
                description = client.describe_collection(self.s.milvus_collection)
                dimension = next((int(x['params']['dim']) for x in description['fields'] if x['name'] == 'vector'), None)
                if dimension != self.s.embedding_dim:
                    raise ValueError('Milvus 集合维度不匹配，请更换集合名称并重建索引')
                client.load_collection(self.s.milvus_collection)
                self._client = client
        return self._client

    def index_document(self, doc, chunks):
        if self.s.app_mode == 'demo':
            return
        texts = [f'{doc.title}\n{x.section}\n{x.text}' for x in chunks]
        # Publication is outside the LangGraph retry policy. Bound embedding
        # retries here; authentication/validation errors must fail immediately.
        for attempt in range(3):
            try:
                vectors = self.models.embed(texts)
                break
            except Exception as exc:
                if not transient(exc) or attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        self.milvus().upsert(collection_name=self.s.milvus_collection, data=[
            {'id': x.id, 'document_id': doc.id, 'vector': vector, 'model': self.s.embedding_model}
            for x, vector in zip(chunks, vectors)
        ])
        self.milvus().flush(self.s.milvus_collection)

    def search(self, query, as_of, limit=6, codes=None):
        with Session() as db:
            pairs = db.execute(select(Chunk, Document).join(Document, Chunk.document_id == Document.id).where(
                Document.status == 'published', Document.effective_from <= as_of,
                (Document.effective_to.is_(None)) | (Document.effective_to > as_of),
            )).all()
            if codes:
                pairs = [(c, d) for c, d in pairs if d.code in codes]
            candidates = {c.id: source(c, d) for c, d in pairs}
        if not candidates:
            return []
        rows = list(candidates.values())
        # BM25L keeps IDF positive for tiny filtered corpora (e.g. one policy
        # version), avoiding Okapi's negative/zero IDF suppressing all matches.
        bm25 = BM25L([tokenize(f'{x["code"]} {x["title"]} {x["section"]} {x["text"]}') for x in rows])
        scores = bm25.get_scores(tokenize(query))
        lexical = sorted([(x['id'], float(v)) for x, v in zip(rows, scores) if v > 0], key=lambda x: x[1], reverse=True)[:12]
        ranking = {}
        for rank, (id_, score) in enumerate(lexical):
            ranking[id_] = 1 / (60 + rank + 1)
        if self.s.app_mode == 'live':
            vector = self.models.embed([query])[0]
            doc_ids = sorted({x['document_id'] for x in rows})
            import json
            # Visibility filter uses IDs derived from the transactional DB, not
            # user-supplied expressions. Recheck results before forming evidence.
            expression = f'document_id in {json.dumps(doc_ids)} and model == {json.dumps(self.s.embedding_model)}'
            hits = self.milvus().search(collection_name=self.s.milvus_collection, data=[vector],
                anns_field='vector', filter=expression, limit=12, search_params={'metric_type': 'COSINE', 'params': {'ef': self.s.hnsw_ef}})[0]
            for rank, hit in enumerate(hits):
                if hit['id'] in candidates:
                    ranking[hit['id']] = ranking.get(hit['id'], 0) + 1 / (60 + rank + 1)
        merged = [dict(candidates[k], fusion_score=v) for k, v in sorted(ranking.items(), key=lambda x: x[1], reverse=True)[:16]]
        if self.s.app_mode == 'live':
            merged = self.reranker.rerank(query, merged)
        return merged[:limit]

    def expand(self, selected, max_chars=12000):
        # Return immediate sibling sections when they fit; never fetch an
        # unrestricted full corpus into the model context.
        output, used = [], set()
        for item in selected:
            if item['id'] not in used and sum(len(x['text']) for x in output) + len(item['text']) <= max_chars:
                output.append(item)
                used.add(item['id'])
        with Session() as db:
            for item in list(output):
                doc = db.get(Document, item['document_id'])
                for chunk in db.scalars(select(Chunk).where(Chunk.document_id == doc.id)).all():
                    if chunk.id not in used and sum(len(x['text']) for x in output) + len(chunk.text) <= max_chars:
                        output.append(source(chunk, doc))
                        used.add(chunk.id)
        return output
