import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import settings


def tokens(text):
    # Deliberately conservative estimate for Chinese; actual provider usage is
    # recorded separately. This is a configurable budget, not a tokenizer claim.
    return max(1, len(text.encode('utf-8')) // 2)


def topic_for(text):
    if any(w in text for w in ('促销', '折扣', '活动', '预算', '投放')):
        return '活动与审批'
    if any(w in text for w in ('入职', '学习', '培训', '第一周')):
        return '入职学习'
    if any(w in text for w in ('周报', '指标', '报表', '转化', 'GMV')):
        return '经营分析'
    if any(w in text for w in ('项目', '交接', '启动', '合同')):
        return '项目协作'
    return ''


def demo_understand(question, memory, profile, message_id):
    memory = json.loads(json.dumps(memory or {}))
    current = memory.get('current', {})
    archives = memory.get('archives', {})
    topic = topic_for(question) or current.get('topic', '一般咨询')
    if topic != current.get('topic'):
        if current.get('topic'):
            archives[current['topic']] = current
        current = dict(archives.get(topic, {}))
    subject = current.get('subject', {'name': '本人', 'position': profile['position'], 'probation': profile['probation']})
    if any(x in question for x in ('帮', '同事', '客服同事')):
        subject = {'name': '同事', 'position': '客服专员' if '客服' in question else profile['position'], 'probation': None}
    if any(x in question for x in ('我自己', '回到我', '问我本人')):
        subject = {'name': '本人', 'position': profile['position'], 'probation': profile['probation']}
    if any(x in question for x in ('更正', '纠正', '说错')):
        if '运营' in question and '不是运营' not in question:
            subject['position'] = '店铺运营专员'
        elif '客服' in question and '不是客服' not in question:
            subject['position'] = '客服专员'
    if '转正' in question or '正式员工' in question:
        subject['probation'] = not ('已转正' in question or '已经转正' in question or '正式员工' in question)
    elif '试用期' in question:
        subject['probation'] = True
    match = re.search(r'(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})', question)
    as_of = '-'.join([match[1], match[2].zfill(2), match[3].zfill(2)]) if match else current.get('as_of', '')
    if not match and any(x in question for x in ('现在', '当前', '今天', '目前')):
        as_of = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    current.update(topic=topic, subject=subject, as_of=as_of, source_message_id=message_id)
    current['facts'] = (current.get('facts', []) + [{'message_id': message_id, 'text': question}])[-12:]
    memory.update(current=current, archives=archives)
    query = question + ' ' + topic + ' ' + subject['position']
    if subject['probation'] is True:
        query += ' 试用期'
    return {'memory': memory, 'query': query, 'route': 'multi' if any(x in question for x in ('找谁', '审批', '能否', '能不能', '能直接', '流程', '哪些部门')) else 'simple', 'as_of': as_of}


def recall(history, question, recent_ids):
    # Old raw messages are scoped to one owned conversation by the service.
    words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z]+', question))
    scored = []
    for x in history:
        if x['id'] in recent_ids or x['role'] != 'user':
            continue
        score = sum(w in x['content'] for w in words)
        if score or any(w in question for w in ('前面', '之前', '刚才', '他', '那')):
            scored.append((score, x))
    return [x for _, x in sorted(scored, key=lambda x: (x[0], x[1]['id']), reverse=True)[:3]]


def context(history, summary, question):
    from .compression import CONDITIONS, paragraphs, compression_events
    s = settings()
    recent = history[-s.recent_rounds * 2:]
    recalled = recall(history, question, {x['id'] for x in recent})
    payload = {'summary': summary, 'recent': recent, 'recalled_originals': recalled}
    protected = []

    def preserve(messages):
        for m in messages:
            if m['role'] == 'user':
                parts = [p for p in paragraphs(m['content']) if CONDITIONS.search(p)]
                if parts:
                    protected.append({'message_id': m['id'], 'text': '\n\n'.join(parts)})
        if protected:
            payload['protected_history'] = protected
    # Never silently truncate the current user question or structured facts.
    omitted = []
    while tokens(json.dumps(payload, ensure_ascii=False)) > s.memory_token_budget and payload['recent']:
        removed = payload['recent'][:2]
        preserve(removed)
        omitted.extend(x['id'] for x in removed)
        payload['recent'] = payload['recent'][2:]
    while tokens(json.dumps(payload, ensure_ascii=False)) > s.memory_token_budget and payload['recalled_originals']:
        preserve([payload['recalled_originals'].pop()])
    if tokens(json.dumps(payload, ensure_ascii=False)) > s.memory_token_budget:
        parts = [p for p in paragraphs(summary) if CONDITIONS.search(p)]
        payload['summary'] = '\n\n'.join(parts)
    if tokens(json.dumps(payload, ensure_ascii=False)) > s.memory_token_budget:
        raise ValueError('关键历史条件超过记忆预算，请缩小问题范围或提高 MEMORY_TOKEN_BUDGET；原始记录已保留')
    payload['omitted_message_ids'] = omitted
    collector = compression_events.get()
    if collector is not None:
        collector.append({'stage': 'memory_context', 'detail': {
            'memory_budget': s.memory_token_budget, 'recent_rounds': s.recent_rounds,
            'omitted_message_ids': omitted, 'recalled_message_ids': [m['id'] for m in payload['recalled_originals']],
            'protected_message_ids': [m['message_id'] for m in protected]}})
    return payload


def summarize(models, history, old_summary, summary_until):
    from .compression import CONDITIONS, paragraphs
    s = settings()
    unsummarized = [x for x in history if x['id'] > summary_until]
    old = unsummarized[:-s.recent_rounds * 2]
    backlog = old_summary + '\n'.join(x['content'] for x in unsummarized)
    if not old or tokens(backlog) < s.memory_token_budget * s.summary_trigger_ratio:
        return old_summary, summary_until, False
    required = list(dict.fromkeys(
        [p for p in paragraphs(old_summary) if CONDITIONS.search(p)] +
        [p for m in old if m['role'] == 'user' for p in paragraphs(m['content']) if CONDITIONS.search(p)]))
    if s.app_mode == 'demo':
        # A transparent extractive digest, not an LLM summary in demo mode.
        lines = [f'消息{x["id"]}：{x["content"][:160]}' for x in old if x['role'] == 'user']
        summary = (old_summary + '\n' + '\n'.join(lines))[-1800:]
        summary += '\n\n' + '\n\n'.join(p for p in required if p not in summary)
    else:
        result = models.json('将旧对话压缩为摘要。保留问题对象、日期、限制、纠正、未解决事项和原始消息编号；不要把助手未经证实的说法变为事实。返回 {"summary": "..."}，不超过800字。',
                             {'previous_summary': old_summary, 'messages': old,
                              'protected_statements': required, 'requirement': 'protected_statements 中的条目必须逐字保留在摘要里，不能改写或省略'})
        summary = result.get('summary')
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError('摘要格式错误')
    if len(summary) > 2400 or any(p not in summary for p in required):
        raise ValueError('摘要未通过关键条件校验，保留旧摘要与原始消息')
    return summary, old[-1]['id'], True
