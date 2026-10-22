"""Chunking ablation on frozen onboarding data; no production index changes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from backend.app.config import settings
from backend.app.providers import Models
from backend.app.reranker import LocalReranker
from backend.app.memory import tokens
from evaluation.chunking import body_and_sections, fixed_spans, recursive_spans, sentence_spans, semantic_spans, section_for
from scripts.benchmark_retrieval import digest, load_corpus, retry, retrieval_phase, save


class CachedEmbeddings:
    def __init__(self,model):
        self.model=model
        s=settings()
        self.path=Path('data/evaluation-cache') / ('chunk-v1-'+digest([s.embedding_model,s.embedding_dim,s.siliconflow_base_url])+'.json')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.cache=json.loads(self.path.read_text()) if self.path.exists() else {}
    def embed(self,texts):
        keys=[digest(t) for t in texts]
        missing={digest(t):t for t in texts if digest(t) not in self.cache}
        if missing:
            vectors=retry(lambda:self.model.embed(list(missing.values())))
            self.cache.update(zip(missing,vectors))
            self.path.write_text(json.dumps(self.cache),encoding='utf-8')
        return [self.cache[k] for k in keys]


class CachedReranker:
    def __init__(self):
        self.base=LocalReranker()
        s=settings()
        self.path=Path('data/evaluation-cache') / ('chunk-scores-'+digest([s.rerank_model,s.rerank_precision,s.rerank_device,s.rerank_max_length])+'.json')
        self.cache=json.loads(self.path.read_text()) if self.path.exists() else {}
    def load(self):
        self.base.load()
    def rerank(self,query,rows,top_n=6):
        pairs=[[query,f'{r["title"]}\n{r["section"]}\n{r["text"]}'] for r in rows]
        keys=[digest(p) for p in pairs]
        missing={digest(p):p for p in pairs if digest(p) not in self.cache}
        if missing:
            missing_pairs=list(missing.values())
            scores=[]
            for i in range(0,len(missing_pairs),settings().rerank_batch_size):
                scores.extend(self.base.score_pairs(missing_pairs[i:i+settings().rerank_batch_size]))
            self.cache.update(zip(missing,scores))
            self.path.write_text(json.dumps(self.cache),encoding='utf-8')
        return sorted([dict(r,rerank_score=self.cache[k]) for r,k in zip(rows,keys)],key=lambda r:r['rerank_score'],reverse=True)[:top_n]


def normalize(text):
    return re.sub(r'\s+','',text)


def make_rows(documents,strategy,cap,threshold):
    output=[]
    for entry in documents:
        text,sections=entry['body'],entry['sections']
        if strategy=='fixed':
            spans=fixed_spans(text,cap)
        elif strategy=='recursive':
            spans=recursive_spans(text,cap)
        else:
            spans=semantic_spans(text,entry['units'],entry['similarities'],threshold,cap)
        for n,(a,b) in enumerate(spans):
            content=text[a:b].strip()
            if not content:
                continue
            output.append({'id':hashlib.sha256(f'{strategy}:{cap}:{entry["key"]}:{n}'.encode()).hexdigest()[:32],
                'document_id':entry['key'],'code':entry['code'],'title':entry['title'],'version':entry['version'],
                'effective_from':entry['effective_from'],'effective_to':entry['effective_to'],
                'section':section_for(a,b,sections),'text':content,'location':f'字符{a}-{b}',
                'refs':sorted(set(re.findall(r'\b[A-Z]+-\d{3}\b',content)))})
    return output


def supplement_metrics(report,rows,documents):
    quote_scores=[]
    evidence_tokens=[]
    for result in report['results']:
        case=result['case']
        if not case['answerable']:
            continue
        chosen=result['retrieved']['hybrid_rerank']
        evidence_tokens.append(tokens('\n\n'.join(r['text'] for r in chosen)))
        gold=case['gold']
        score={f'quote_recall_at_{k}':sum(any(r['document_id']==g['document'] and normalize(g['quote']) in normalize(r['text'])
                 for r in chosen[:k]) for g in gold)/len(gold) for k in (3,6)}
        result['quote_scores']=score
        quote_scores.append(score)
    protected=[]
    for doc in documents:
        for sentence in re.split(r'(?<=[。！？；])',doc['body']):
            sentence=sentence.strip()
            if len(sentence)>5 and re.search(r'不能|不得|必须|需导师|仅限|除非|不承诺|不能替代',sentence):
                protected.append((doc['key'],sentence))
    complete=sum(any(r['document_id']==key and normalize(sentence) in normalize(r['text']) for r in rows) for key,sentence in protected)
    broken=[{'document_id':key,'clause':sentence,'chunks':[r['text'] for r in rows if r['document_id']==key]}
        for key,sentence in protected if not any(r['document_id']==key and normalize(sentence) in normalize(r['text']) for r in rows)]
    lengths=[len(r['text']) for r in rows]
    return {**report['retrieval_summary']['hybrid_rerank'],
        **{k:float(np.mean([s[k] for s in quote_scores])) for k in quote_scores[0]},
        'chunks':len(rows),'mean_chars':float(np.mean(lengths)),'max_chars':max(lengths),
        'protected_sentences':len(protected),'intact_protected_sentences':complete,
        'protected_integrity':complete/len(protected) if protected else None,
        'broken_clause_examples':broken,
        'mean_evidence_body_tokens_estimate':float(np.mean(evidence_tokens)),
        'over_cap_chunks':sum(n>report['chunking_config']['cap'] for n in lengths)}


def render(report,dest):
    save(report,dest)
    lines=['# 分块策略对比实验','',
        '固定18份模拟文档/20个版本、48条正例及6条负例。比较完整混合检索+BGE重排后的检索与证据完整性，不调用生成或评审模型。',
        '分块上限2200为当前配置；128为预先指定的短块边界压力实验，不是根据分数挑选的最优参数。','',
        '| 字符目标 | 分块策略 | 块数 | 平均长度 | P@1 | 文档Recall@3 | MRR@6 | 原文证据@6 | 限制条款完整率 | 证据正文估算Token |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for variant in report['variants']:
        s=variant['summary']
        lines.append(f'| {variant["cap"]} | {variant["strategy"]} | {s["chunks"]} | {s["mean_chars"]:.1f} | {s["p_at_1"]:.1%} | '
            f'{s["recall_at_3"]:.1%} | {s["mrr_at_6"]:.3f} | {s["quote_recall_at_6"]:.1%} | {s["protected_integrity"]:.1%} | '
            f'{s.get("mean_evidence_body_tokens_estimate",0):.1f} |')
    lines+=['','## 控制变量与口径','',
        '- 每组复用相同文档标题及可定位的章节元数据、日期过滤、Qwen3-Embedding-0.6B、COSINE HNSW（M=16、efConstruction=128、ef=64）、BM25L、RRF k=60、分支12/融合16/最终6片段。各组只改变分块边界策略与指定长度。',
        '- heading调用真实add_chunks函数；fixed按字符硬切；recursive依次按段落、句子、换行、空白和字符回退切分；semantic按相邻句向量相似度低点分块，超长语义组用同一递归方法满足字符上限。各组不加overlap。',
        '- heading的长度为软目标，保留整行，可能超过上限；其他三种执行硬上限。跨章节的替代分块附覆盖章节名称，避免只给heading额外标题信息。标题行不重复进入正文。',
        f'- semantic阈值为全语料相邻句余弦相似度的25分位数（{report["semantic_threshold"]:.4f}），最短语义合并长度64字符；只依赖语料，不使用问题或相关标签调参。这是明确指定的一种语义分块实现，不能代表全部语义分块算法。',
        '- 文档指标按48条正例宏平均；原文证据@6表示指定gold原文在前6个片段中被某个单片段完整包含的比例，不把两个残片拼接当作完整命中。',
        '- 限制条款完整率：冻结语料中含不能/不得/必须等词的完整句子，有多少能在同一片段中找到。它是自动边界诊断，不是人工标注的所有关键条件。',
        '- 证据正文估算Token是48条正例前6片段正文的平均UTF-8估算量，不含提示词、来源元数据和历史，不等于真实模型输入用量或费用。',
        '- 语料整体较短，2200字符组不一定触发长度切分；fixed和recursive相同结果可能来自完全一致的边界，不能认为大规模长文档下二者等效。128压力实验不能直接外推为生产表现。',
        '- Embedding和本地重排分数按模型参数及完整文本缓存以复用完全相同计算，不复用其他策略的排名。不同缓存命中导致时延不可公平比较，本轮不报告速度结论。',
        '- 当前不改动线上分块/索引；独立临时Milvus集合按组创建并删除。各组原始排名、证据、分块和指纹保留在data/chunking-ablation目录。',
        '- 本次结果只适用于模拟小样本，未做独立人工复核或统计显著性检验，不根据某一项分数自动替换当前分块。','']
    lines.insert(-1,'- 所有分块方法统一使用LF换行，避免CRLF计数和预处理差异。')
    examples=[(v,x) for v in report['variants'] for x in v['summary'].get('broken_clause_examples',[])][:3]
    if examples:
        lines+=['','## 条款边界诊断示例','']
        for variant,item in examples:
            lines.append(f'- {variant["strategy"]} / {variant["cap"]} / {item["document_id"]}：`{item["clause"]}` 未在任一单片段中完整保留；原始分块见对应chunks.json。')
    dest.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--caps',type=int,nargs='+',default=[2200,128])
    args=parser.parse_args()
    if any(n<64 for n in args.caps):
        parser.error('cap must be >=64')
    root=Path('evaluation/retrieval_v1')
    dataset=json.loads((root/'dataset.json').read_text(encoding='utf-8'))
    documents=[]
    for entry in dataset['corpus']:
        raw=(root/entry['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise ValueError('Corpus changed')
        body,sections=body_and_sections(raw.decode('utf-8'))
        documents.append({**entry,'body':body,'sections':sections,'units':sentence_spans(body)})
    dest=Path('data/chunking-ablation/report.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    models=Models()
    cache=CachedEmbeddings(models)
    reranker=CachedReranker()
    report={'dataset_fingerprint':digest(dataset),'caps':args.caps,'variants':[]}
    try:
        all_similarities=[]
        for doc in documents:
            vectors=cache.embed([doc['body'][a:b].strip() for a,b in doc['units']])
            array=np.asarray(vectors)
            array=array/np.maximum(np.linalg.norm(array,axis=1,keepdims=True),1e-12)
            sims=np.sum(array[:-1]*array[1:],axis=1).tolist()
            doc['similarities']=sims
            all_similarities+=sims
        threshold=float(np.percentile(all_similarities,25))
        report['semantic_threshold']=threshold
        report['document_body_chars']={'min':min(len(d['body']) for d in documents),'max':max(len(d['body']) for d in documents)}
        for cap in args.caps:
            for strategy in ('heading','fixed','recursive','semantic'):
                name=f'{strategy}-{cap}'
                print('BEGIN '+name,flush=True)
                rows=load_corpus(root,dataset,max_chars=cap) if strategy=='heading' else make_rows(documents,strategy,cap,threshold)
                row_file=dest.parent/(name+'-chunks.json')
                row_file.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
                output=dest.parent/(name+'.json')
                identity=digest(rows)
                if output.exists():
                    prior=json.loads(output.read_text(encoding='utf-8'))
                else:
                    prior={}
                expected={'embedding':settings().embedding_model,'rerank':settings().rerank_model,
                          'rerank_precision':settings().rerank_precision}
                compatible=all(prior.get('config',{}).get(k)==v for k,v in expected.items())
                if compatible and prior.get('rows_fingerprint')==identity and prior.get('dataset_fingerprint')==digest(dataset) and len(prior.get('results',[]))==len(dataset['cases']):
                    result=prior
                else:
                    result=retrieval_phase(SimpleNamespace(output=output),dataset,rows,cache,reranker=reranker)
                result['chunking_config']={'strategy':strategy,'cap':cap,'overlap':0}
                result['rows_fingerprint']=identity
                summary=supplement_metrics(result,rows,documents)
                save(result,output)
                report['variants'].append({'strategy':strategy,'cap':cap,'summary':summary,'detail_file':str(output)})
                render(report,dest)
                print('DONE '+name+' '+json.dumps(summary),flush=True)
    finally:
        models.close()


if __name__=='__main__':
    main()
