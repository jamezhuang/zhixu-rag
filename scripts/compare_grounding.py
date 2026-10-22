"""Compare all retrieval methods with the same generation/judging rubric.

Uses saved retrieval outputs, never reindexes or edits the business database.
Existing BGE answers are fixed; only missing/failed evaluations make API calls.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

from backend.app.config import settings
from backend.app.providers import Models
from evaluation.metrics import grounding_metrics
from scripts.benchmark_retrieval import METHODS, digest, evaluate_answer, save


def summarize(report, dest):
    summaries = {}
    for method in METHODS:
        evaluations = [r['evaluations'][method] for r in report['results'] if method in r['evaluations']]
        summary = grounding_metrics(evaluations)
        summary['pending'] = len(report['results']) - len(evaluations)
        for category, answerable in [('answerable', True), ('negative', False)]:
            entries = [r['evaluations'][method] for r in report['results']
                       if r['case']['answerable'] is answerable and method in r['evaluations']]
            valid = [e for e in entries if e['status'] == 'ok']
            summary[category] = {**grounding_metrics(entries),
                                 'abstained_answers': sum(e['abstained'] for e in valid)}
        summaries[method] = summary
    report['summary'] = summaries
    save(report, dest)
    pct = lambda v: f'{v:.2%}' if v is not None else 'N/A'
    lines = ['# 忠实度与幻觉率横向对比', '',
        '固定同一批 54 条问题（48 条正例、6 条负例），使用各方案对应的固定检索输出、相同 DeepSeek 生成提示词与 v1.2 评审规则。',
        '已有混合检索加重排的 54 条回答和评审复用，另外三组补测；不更换标准标签，不按结果筛选回答。', '',
        '| 方案 | 有效评审/54 | 可核验结论 | 忠实度 | 结论级幻觉率 | 无法判定 | 含幻觉回答比例 | 正例拒答 | 负例拒答 | 无事实回答 | 评审失败 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method,s in summaries.items():
        lines.append(f'| {method} | {s["evaluated_answers"]} | {s["checkable_claims"]} | {pct(s["faithfulness_micro"])} | '
            f'{pct(s["hallucination_micro"])} | {pct(s["uncertain_ratio"])} | {pct(s["answer_hallucination_rate"])} | '
            f'{s["answerable"]["abstained_answers"]}/{s["answerable"]["evaluated_answers"]} | '
            f'{s["negative"]["abstained_answers"]}/{s["negative"]["evaluated_answers"]} | '
            f'{s["answers_without_checkable_claims"]} | {s["failed_evaluations"]} |')
    lines += ['', '## 正例单独统计', '', '| 方案 | 可核验结论 | 忠实度 | 幻觉率 |', '|---|---:|---:|---:|']
    for method,s in summaries.items():
        a = s['answerable']
        lines.append(f'| {method} | {a["checkable_claims"]} | {pct(a["faithfulness_micro"])} | {pct(a["hallucination_micro"])} |')
    lines += ['', '## 对应的检索覆盖', '', '| 方案 | P@1 | Recall@3 | 完整证据覆盖@3 |', '|---|---:|---:|---:|']
    for method,s in report.get('retrieval_summary',{}).items():
        lines.append(f'| {method} | {pct(s["p_at_1"])} | {pct(s["recall_at_3"])} | {pct(s["all_gold_at_3"])} |')
    lines += ['', '## 口径及限制', '',
        '- 忠实度为 supported / 全部可核验结论；幻觉率为 (unsupported + contradicted) / 全部可核验结论，uncertain 单列。不是与标准答案比对的正确率。',
        '- 评审对象是回答的所有可核验结论；单独无法确认的非事实声明由固定规则剔除并保留审计。无事实回答不获得 100% 忠实度。',
        '- 每个方案都取其已保存的最多六个检索片段，日期过滤、生成和评审规则相同；数据集中的标准标签不输入生成/评审模型。',
        '- API 生成与评审为同型号 DeepSeek，存在随机性和共同偏差；本次为每方案一次实验，未独立人工复核，不能证明统计显著提升。',
        '- 忠实度仅说明回答是否有检索依据。检索遗漏信息后回答可能仍忠实但不完整，须结合 Recall@3、完整证据覆盖率与拒答数量解读。',
        '- 当前未执行 LangGraph 改写、多跳、答案核验修正和会话记忆，结果只对应单轮检索加直接生成组件。',
        '- 非零评审失败必须单列，不能当成正确或无幻觉；断点续跑保留原回答，只重试未完成评审。',
        '- 逐题回答、支持原文、评审结论、失败及重试历史保存在同名 JSON。', '']
    lines.insert(-1, '- 结论片段只允许去除末尾标点后与回答逐字对齐，不能修补内部措辞；原片段与对齐标记保留，证据引用仍严格校验。')
    dest.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('data/retrieval-evaluation.json'))
    parser.add_argument('--output',type=Path,default=Path('data/grounding-comparison.json'))
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('workers must be 1..4')
    source = json.loads(args.source.read_text(encoding='utf-8'))
    expected_version = 'v1.2-metadata-disclaimer'
    for row in source['results']:
        if row['grounding'].get('evaluator_version') != expected_version:
            raise ValueError('Source evaluations must use the same v1.2 rubric')
        used = {m['model'] for m in row['grounding'].get('metrics',[]) if m.get('endpoint') == 'chat/completions'}
        if used and used != {settings().deepseek_model}:
            raise ValueError('Generation/judge model differs from source experiment')
    identity = digest({'dataset':source['dataset_fingerprint'],
                       'retrieval':[{'case':r['case'],'retrieved':r['retrieved']} for r in source['results']]})
    if args.output.exists():
        report = json.loads(args.output.read_text(encoding='utf-8'))
        if report['source_fingerprint'] != identity:
            raise ValueError('Source changed; use a new output path for a new experiment')
    else:
        report = {'source_fingerprint':identity,'created_at':datetime.now(timezone.utc).isoformat(),
            'generation_and_judge_model':settings().deepseek_model,'evaluator_version':'v1.2-metadata-disclaimer',
            'results':[{'case':r['case'],'retrieved':r['retrieved'],
                        'evaluations':{'hybrid_rerank':r['grounding']}} for r in source['results']]}
    pending = [(r,m) for r in report['results'] for m in METHODS
               if r['evaluations'].get(m,{}).get('status') != 'ok']
    report['retrieval_summary'] = source['retrieval_summary']
    summarize(report,args.output)
    print(f'{len(pending)} pending method/question evaluations; existing answers will be preserved.',flush=True)
    models = Models()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for row,method in pending:
                payload = {'case':row['case'],'retrieved':row['retrieved'], 'grounding':row['evaluations'].get(method,{})}
                futures[pool.submit(evaluate_answer,payload,models,method)] = row,method
            for number,future in enumerate(as_completed(futures),1):
                row,method = futures[future]
                if method in row['evaluations']:
                    row.setdefault('previous_attempts',{}).setdefault(method,[]).append(row['evaluations'][method])
                row['evaluations'][method] = future.result()
                summarize(report,args.output)
                print(f'{number}/{len(pending)} {row["case"]["id"]} {method}: {row["evaluations"][method]["status"]}',flush=True)
        print(args.output.with_suffix('.md'),flush=True)
    finally:
        models.close()


if __name__ == '__main__':
    main()
