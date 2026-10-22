"""Create two reviewable demo conversations through the real local API."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path

import httpx

from backend.app.config import settings


CASES = [
    ('multi_hop', '我是试用期运营。请结合试用期授权、活动审批和临时变更流程说明：客户临时改折扣时，我能否直接确认？导师复核后还需谁审批和确认，执行后同步哪些岗位？'),
    ('retry_limit', '2025年12月31日，试用期运营能否独立确认促销折扣？请查找当时有效的权限制度和活动审批流程，给出具体条款。'),
]


def run_case(case):
    name, question = case[:2]
    with httpx.Client(base_url='http://127.0.0.1:8000/api', timeout=600) as client:
        response = client.post('/auth/login',json={'username':'admin','password':settings().admin_password})
        response.raise_for_status()
        client.headers['Authorization'] = 'Bearer ' + response.json()['token']
        if len(case) > 2:
            conv, run_id = case[2:4]
        else:
            response = client.post('/conversations')
            response.raise_for_status()
            conv = response.json()['id']
            response = client.post(f'/conversations/{conv}/requests',json={'question':question})
            response.raise_for_status()
            run_id = response.json()['id']
        response = client.post(f'/runs/{run_id}/execute')
        response.raise_for_status()
        result = response.json()
        response = client.get('/admin/runs')
        response.raise_for_status()
        saved = next(r for r in response.json() if r['id'] == run_id)
        events = [e for e in saved['events'] if e['stage'] in
                  ('understand','retrieve','hop','assess','rewrite','supplement','generate','verify','repair','failed')]
        return {'case':name,'question':question,'conversation_id':conv,'run_id':run_id,
                'status':result['status'],'error':result.get('error'),
                'answer':result.get('result',{}).get('answer'), 'events':events,
                'actual_second_hop':any(e['stage']=='hop' and e['detail'].get('dependency_refs') for e in events),
                'rewrite_rounds':sum(e['stage']=='rewrite' for e in events),
                'supplement_rounds':sum(e['stage']=='supplement' for e in events)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true',help='Resume stored runs without creating new conversations')
    args = parser.parse_args()
    path = Path('data/langgraph-route-demo.json')
    cases = CASES
    if args.resume:
        original = path.read_text(encoding='utf-8')
        path.with_name('langgraph-route-demo-before-resume.json').write_text(original,encoding='utf-8')
        cases = [(r['case'],r['question'],r['conversation_id'],r['run_id']) for r in json.loads(original)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run_case,cases))
    path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in results:
        print(json.dumps({k:r[k] for k in ('case','status','actual_second_hop','rewrite_rounds','supplement_rounds')},ensure_ascii=True))
    print(path)


if __name__ == '__main__':
    main()
