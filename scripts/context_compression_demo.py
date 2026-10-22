"""Reproducible synthetic long-session compression; no model/API calls."""
import json
from pathlib import Path

from backend.app.compression import compact_payload
from backend.app.config import Settings


def main():
    background = '\n\n'.join(f'背景段落{i}：' + '普通经营背景说明。' * 18 for i in range(30))
    payload = {
        'question': '试用期运营可以直接确认折扣吗？', 'as_of': '2026-10-21',
        'memory': {'current': {'subject': {'name': '本人', 'position': '运营', 'probation': True},
                               'facts': [{'message_id': 1, 'text': '更正：目前仍在试用期。'}]}},
        'context': {'summary': '尚未解决：确认审批权限。\n\n' + background,
                    'recent': [{'id': i, 'role': 'user' if i % 2 else 'assistant',
                                'content': background} for i in range(1, 9)]},
        'evidence': [{'id': 'simulated-chunk', 'document_id': 'simulated-doc', 'code': 'AUTH-001',
            'version': '2.0', 'citation': 1, 'effective_from': '2026-07-01', 'effective_to': None,
            'text': background + '\n\n试用期运营不能独立确认折扣。\n\n导师复核不能替代主管审批。'}]}
    s = Settings(_env_file=None, context_token_budget=16000, deepseek_max_tokens=2400, context_safety_tokens=800)
    result, report = compact_payload('仅根据有效证据回答企业制度问题。', payload, s)
    report.update(dataset='synthetic long session; no model/API calls',
        conditions_preserved=result['memory'] == payload['memory'],
        source_preserved=all(result['evidence'][0][k] == payload['evidence'][0][k]
                             for k in ('id', 'document_id', 'version', 'citation', 'effective_from')),
        exceptions_preserved=all(t in result['evidence'][0]['text'] for t in
                                ('试用期运营不能独立确认折扣。', '导师复核不能替代主管审批。')),
        reduction_ratio=round(1 - report['after_tokens_estimate'] / report['before_tokens_estimate'], 4))
    dest = Path('data/context-compression-report.json')
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
