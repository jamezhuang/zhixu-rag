"""Run reviewable business cases. Does not invent an automated accuracy score."""
import argparse
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000/api')
    parser.add_argument('--password', default='Welcome@2026')
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--output', default='data/evaluation.json')
    args = parser.parse_args()
    cases = json.loads(Path('evaluation/cases.json').read_text(encoding='utf-8'))
    results = []
    with httpx.Client(timeout=600) as client:
        for case in cases[:args.limit]:
            auth = client.post(args.url + '/auth/login', json={'username': case['account'], 'password': args.password})
            auth.raise_for_status()
            headers = {'Authorization': f'Bearer {auth.json()["token"]}'}
            conv = None
            turns = []
            started = time.perf_counter()
            for question in case['turns']:
                if conv is None or case.get('new_session_between'):
                    response = client.post(args.url + '/conversations', headers=headers)
                    response.raise_for_status()
                    conv = response.json()['id']
                response = client.post(f'{args.url}/conversations/{conv}/requests', headers=headers, json={'question': question})
                response.raise_for_status()
                response = client.post(f'{args.url}/runs/{response.json()["id"]}/execute', headers=headers)
                response.raise_for_status()
                turns.append(response.json())
                if turns[-1]['status'] != 'completed':
                    break
            final = turns[-1]
            found = sorted({s['code'] for s in final.get('result', {}).get('sources', [])})
            results.append({'case': case, 'turns': turns, 'elapsed_seconds': time.perf_counter() - started,
                            'expected_sources_found': sorted(set(case['sources']) & set(found)),
                            'expected_sources_missing': sorted(set(case['sources']) - set(found)),
                            'human_review': {'correct': None, 'complete': None, 'supported': None, 'notes': ''}})
            print(f'{case["id"]}: {final["status"]}; evidence {found}', flush=True)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {len(results)} cases to {args.output}. Human review required.')


if __name__ == '__main__':
    main()
