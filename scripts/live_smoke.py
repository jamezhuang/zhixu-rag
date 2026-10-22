"""Real PG + Milvus + API contract fixtures, never paid model calls.

Run: python -m scripts.live_smoke
Creates an isolated UUID schema/collection and removes only those resources.
"""
import hashlib
import json
import os
import tempfile
import uuid
import argparse

import httpx
import psycopg
from psycopg import sql


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-reranker', action='store_true', help='Use the downloaded local BGE model instead of ranking fixtures')
    args = parser.parse_args()
    schema = 'zhixu_smoke_' + uuid.uuid4().hex[:12]
    uri = os.getenv('TEST_POSTGRES_URI', 'postgresql://zhixu:zhixu_local_only@127.0.0.1:55432/zhixu')
    with psycopg.connect(uri, autocommit=True, connect_timeout=5) as conn:
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
    os.environ.update(APP_MODE='live', DATABASE_URL=uri.replace('postgresql://', 'postgresql+psycopg://', 1) + f'?options=-csearch_path%3D{schema}',
                      DATA_DIR=tempfile.mkdtemp(prefix='zhixu-live-'), MILVUS_COLLECTION=schema, MILVUS_URI='http://127.0.0.1:19531',
                      ADMIN_PASSWORD='SmokeOnly2026!', EMPLOYEE_PASSWORD='SmokeOnly2026!',
                      DEEPSEEK_API_KEY='fixture-not-a-real-key', SILICONFLOW_API_KEY='fixture-not-a-real-key', LANGSMITH_TRACING='false')
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.db import engine
    from pymilvus import MilvusClient

    def fixture(request):
        data = json.loads(request.content)
        if request.url.path.endswith('/embeddings'):
            vectors = []
            for i, text in enumerate(data['input']):
                digest = hashlib.sha256(text.encode()).digest()
                vectors.append({'index': i, 'embedding': [(digest[n % 32] - 127.5) / 127.5 for n in range(data['dimensions'])]})
            return httpx.Response(200, json={'data': vectors, 'usage': {'total_tokens': 10}})
        prompt = data['messages'][0]['content']
        if '理解企业问答' in prompt:
            result = {'query': '试用期运营促销折扣审批', 'route': 'multi', 'as_of': '2026-10-21', 'memory': {'current': {'topic': '活动与审批', 'subject': {'name': '本人', 'position': '店铺运营专员', 'probation': True}}}}
        elif '检查证据是否覆盖' in prompt:
            result = {'checks': {'complete': True, 'supported': True, 'conditions': True, 'version': True}, 'missing': []}
        elif '你是企业知识助手' in prompt:
            result = {'answer': '根据所列制度，试用期员工应按审批流程提交材料。[1]'}
        elif '压缩为摘要' in prompt:
            result = {'summary': '这是测试摘要，原始记录仍保留。'}
        else:
            result = {'answer_supported': True, 'issues': []}
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(result, ensure_ascii=False)}}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}})

    try:
        with TestClient(app) as client:
            app.state.models.client.close()
            app.state.models.client = httpx.Client(transport=httpx.MockTransport(fixture))
            class FixtureReranker:
                def rerank(self, query, candidates):
                    return [dict(x, rerank_score=1 / (i + 1)) for i, x in enumerate(candidates[:6])]
            if not args.real_reranker:
                app.state.retriever.reranker = FixtureReranker()
            login = client.post('/api/auth/login', json={'username': 'admin', 'password': 'SmokeOnly2026!'})
            assert login.status_code == 200, login.text
            headers = {'Authorization': 'Bearer ' + login.json()['token']}
            docs = client.get('/api/documents', headers=headers).json()
            assert len(docs) == 20 and all(x['status'] == 'draft' for x in docs)
            for d in docs:
                if (d['code'], d['version']) in {('HR-002', '2.0'), ('AUTH-001', '2.0'), ('FLOW-006', '1.0')}:
                    response = client.post(f'/api/admin/documents/{d["id"]}/publish', headers=headers)
                    assert response.status_code == 200, response.text
            conv = client.post('/api/conversations', headers=headers).json()['id']
            created = client.post(f'/api/conversations/{conv}/requests', headers=headers, json={'question': '试用期运营可以直接确认折扣吗？'})
            run_id = created.json()['id']
            result = client.post(f'/api/runs/{run_id}/execute', headers=headers).json()
            assert result['status'] == 'completed', result
            assert result['result']['sources']
            assert result['result']['mode'] == 'live'
            trace = next(x for x in client.get('/api/admin/runs', headers=headers).json() if x['id'] == run_id)
            assert trace['metrics'] and any(x['endpoint'] == 'embeddings' for x in trace['metrics'])
            assert any(x['stage'] == 'verify' for x in trace['events'])
            if args.real_reranker:
                assert any(x['endpoint'] == 'local/rerank' and x['status'] == 'ok' for x in trace['metrics'])
            print('PASS: live API lifecycle, PostgreSQL business tables/checkpoints, draft publish, real Milvus HNSW, hybrid retrieval, reflection routing, citations and fixture usage metrics.')
            print('External model HTTP calls used fixtures; this is not a real model-quality or API-availability test.')
            print('Reranker:', 'real local BGE CPU inference' if args.real_reranker else 'fixture')
    finally:
        engine.dispose()
        milvus = MilvusClient('http://127.0.0.1:19531')
        if milvus.has_collection(schema):
            milvus.drop_collection(schema)
        with psycopg.connect(uri, autocommit=True, connect_timeout=5) as conn:
            conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))


if __name__ == '__main__':
    main()
