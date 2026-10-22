"""Verify configured providers and optionally publish the bundled seed corpus.

Uses real API calls and the running local backend. Never prints keys or tokens.
"""
import argparse
import json
from pathlib import Path

import httpx

from backend.app.config import settings
from backend.app.providers import Models
from backend.app.seed import DOCUMENTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish-seed', action='store_true')
    parser.add_argument('--question', default='试用期运营可以直接确认促销折扣吗？不能的话找谁？')
    args = parser.parse_args()
    s = settings()
    report = {'generation': {}, 'embedding': {}, 'published': [], 'question': args.question}
    models = Models()
    try:
        result = models.json('Return exactly this JSON object: {"ok":true}.', {'message': 'ping'})
        if result.get('ok') is not True:
            raise ValueError('Generation JSON contract check failed')
        report['generation'] = {'ok': True, 'model': s.deepseek_model}
        vectors = models.embed(['Employee onboarding knowledge connection check.'])
        report['embedding'] = {'ok': True, 'model': s.embedding_model, 'dimensions': len(vectors[0])}
        print('Generation and embedding API checks passed.', flush=True)
        with httpx.Client(base_url='http://127.0.0.1:8000/api', timeout=600) as client:
            health = client.get('/health')
            health.raise_for_status()
            if health.json()['mode'] != 'live':
                raise ValueError('Backend is not running in live mode')
            login = client.post('/auth/login', json={'username': 'admin', 'password': s.admin_password})
            login.raise_for_status()
            client.headers['Authorization'] = 'Bearer ' + login.json()['token']
            if args.publish_seed:
                response = client.get('/documents')
                response.raise_for_status()
                codes = {x[0] for x in DOCUMENTS}
                for doc in response.json():
                    if doc['code'] not in codes or doc['version'] not in ('1.0', '2.0') or doc['status'] != 'draft':
                        continue
                    published = client.post(f'/admin/documents/{doc["id"]}/publish')
                    published.raise_for_status()
                    report['published'].append(f'{doc["code"]} v{doc["version"]}')
                    print('Published:', report['published'][-1], flush=True)
            response = client.post('/conversations')
            response.raise_for_status()
            conversation = response.json()['id']
            response = client.post(f'/conversations/{conversation}/requests', json={'question': args.question})
            response.raise_for_status()
            run_id = response.json()['id']
            response = client.post(f'/runs/{run_id}/execute')
            response.raise_for_status()
            result = response.json()
            report['run_id'] = run_id
            report['status'] = result['status']
            report['result'] = result['result']
            report['error'] = result['error']
            Path('data').mkdir(exist_ok=True)
            Path('data/live_connection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            if result['status'] != 'completed':
                raise ValueError('Live question failed: ' + result['error'])
            print('Live RAG question completed, sources:', len(result['result']['sources']), flush=True)
            print('Review: data/live_connection.json', flush=True)
    except httpx.HTTPStatusError as exc:
        # Avoid response bodies, request headers, or signed URLs in diagnostics.
        raise SystemExit(f'API request failed: HTTP {exc.response.status_code}') from None
    finally:
        models.close()


if __name__ == '__main__':
    main()
