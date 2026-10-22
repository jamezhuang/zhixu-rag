"""Isolated random-vector ANN experiment; not a business-quality benchmark."""
import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np
from pymilvus import MilvusClient, DataType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uri', default='http://localhost:19531')
    parser.add_argument('--count', type=int, default=5000)
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--queries', type=int, default=50)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--ef', type=int, nargs='+', default=[32, 64, 128])
    args = parser.parse_args()
    if args.count < args.k or min(args.ef) < args.k:
        parser.error('count 和 ef 必须不小于 k')
    rng = np.random.default_rng(42)
    data = rng.normal(size=(args.count, args.dim)).astype('float32')
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    queries = rng.normal(size=(args.queries, args.dim)).astype('float32')
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    client = MilvusClient(args.uri)
    experiment = 'zhixu_bench_' + uuid.uuid4().hex[:12]
    report = {'dataset': 'seeded independent random normalized float32 vectors', 'business_quality': False,
              'config': vars(args), 'experiment': experiment, 'results': [], 'collections': []}
    outputs = {}
    for index_type in ('FLAT', 'HNSW'):
        name = experiment + '_' + index_type.lower()
        report['collections'].append(name)
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field('id', DataType.INT64, is_primary=True)
        schema.add_field('vector', DataType.FLOAT_VECTOR, dim=args.dim)
        client.create_collection(name, schema=schema)
        for offset in range(0, len(data), 500):
            client.insert(name, [{'id': i, 'vector': data[i].tolist()} for i in range(offset, min(offset + 500, len(data)))])
        client.flush(name)
        indexes = client.prepare_index_params()
        indexes.add_index(field_name='vector', index_type=index_type, metric_type='COSINE',
                          params={'M': 16, 'efConstruction': 128} if index_type == 'HNSW' else {})
        start = time.perf_counter()
        client.create_index(name, indexes)
        client.load_collection(name)
        build_ms = (time.perf_counter() - start) * 1000
        for ef in ([None] if index_type == 'FLAT' else args.ef):
            params = {'metric_type': 'COSINE', 'params': {} if ef is None else {'ef': ef}}
            client.search(name, [queries[0].tolist()], anns_field='vector', limit=args.k, search_params=params)
            durations, hits = [], []
            for q in queries:
                start = time.perf_counter()
                result = client.search(name, [q.tolist()], anns_field='vector', limit=args.k, search_params=params)[0]
                durations.append((time.perf_counter() - start) * 1000)
                hits.append({x['id'] for x in result})
            if index_type == 'FLAT':
                outputs['truth'] = hits
            recall = float(np.mean([len(a & b) / args.k for a, b in zip(hits, outputs['truth'])]))
            report['results'].append({'index': index_type, 'ef': ef, 'recall_at_k': recall,
                                      'p50_ms': float(np.percentile(durations, 50)), 'p95_ms': float(np.percentile(durations, 95)),
                                      'build_and_load_ms': build_ms})
        client.release_collection(name)
    report['memory'] = '未自动采集；需按相同条件记录 Milvus 进程 RSS。不可用向量理论大小冒充实际内存。'
    report['cleanup'] = '实验集合保留用于复查；仅手动删除报告列出的 zhixu_bench_* 集合。'
    Path('data').mkdir(exist_ok=True)
    dest = Path('data') / f'{experiment}.json'
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(dest)


if __name__ == '__main__':
    main()
