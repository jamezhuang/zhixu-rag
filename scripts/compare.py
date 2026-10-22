"""Controlled ablations through the admin-only evaluation endpoint."""
import argparse
import json
from pathlib import Path
import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000/api')
    parser.add_argument('--password', default='Admin@2026')
    parser.add_argument('--limit', type=int, default=6)
    parser.add_argument('--variants', nargs='+', default=['full', 'basic', 'recent_only', 'no_reflection'],
                        choices=['full', 'basic', 'recent_only', 'no_reflection'])
    args = parser.parse_args()
    cases = json.loads(Path('evaluation/cases.json').read_text(encoding='utf-8'))
    # Include both cross-document and multi-turn cases before single-step ones.
    cases.sort(key=lambda x: (x['category'] not in ('multi', 'memory'), x['id']))
    results = []
    with httpx.Client(timeout=600) as client:
        login = client.post(args.url + '/auth/login', json={'username': 'admin', 'password': args.password})
        login.raise_for_status()
        headers = {'Authorization': 'Bearer ' + login.json()['token']}
        for case in cases[:args.limit]:
            for variant in args.variants:
                conv, turns = None, []
                for question in case['turns']:
                    body = {'account': case['account'], 'question': question, 'experiment': variant,
                            'conversation_id': None if case.get('new_session_between') else conv}
                    response = client.post(args.url + '/admin/experiments', headers=headers, json=body)
                    response.raise_for_status()
                    run = response.json()
                    conv = run['conversation_id']
                    turns.append(run)
                    if run['status'] != 'completed':
                        break
                results.append({'case': case, 'variant': variant, 'turns': turns,
                                'human_review': {'correct': None, 'complete': None, 'notes': ''}})
                print(case['id'], variant, turns[-1]['status'], flush=True)
                Path('data').mkdir(exist_ok=True)
                Path('data/comparison.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
