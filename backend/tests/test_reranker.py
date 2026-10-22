from backend.app.reranker import LocalReranker
from backend.app.providers import call_metrics


def test_local_reranker_batches_and_orders_without_http(monkeypatch):
    ranker = LocalReranker()
    calls = []
    monkeypatch.setattr(ranker, 'load', lambda: None)
    def scores(pairs):
        calls.append(pairs)
        return [0.9 if '主管审批' in pair[1] else 0.1 for pair in pairs]
    monkeypatch.setattr(ranker, 'score_pairs', scores)
    monkeypatch.setattr(ranker.s, 'rerank_batch_size', 1)
    candidates = [{'id': 'low', 'title': '周报', 'section': '时间', 'text': '周五提交'},
                  {'id': 'high', 'title': '权限', 'section': '折扣', 'text': '需主管审批'}]
    collector = []
    marker = call_metrics.set(collector)
    try:
        result = ranker.rerank('折扣谁审批', candidates)
    finally:
        call_metrics.reset(marker)
    assert result[0]['id'] == 'high'
    assert len(calls) == 2
    assert collector[0]['endpoint'] == 'local/rerank'
    assert collector[0]['usage']['api_tokens'] == 0


def test_missing_local_model_gives_actionable_error(monkeypatch, tmp_path):
    ranker = LocalReranker()
    monkeypatch.setattr(ranker.s, 'rerank_model_path', str(tmp_path / 'absent'))
    try:
        ranker.load()
        assert False
    except ValueError as exc:
        assert 'prepare_reranker' in str(exc)
