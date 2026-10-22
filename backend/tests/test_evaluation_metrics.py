import pytest

from evaluation.metrics import ranking_metrics, grounding_metrics, validate_judgment


def test_rank_metrics_use_gold_denominator_and_deduplicate_versions():
    result = ranking_metrics(['A@1', 'A@1', 'B@1', 'C@1'], {'B@1', 'D@1'})
    assert result == {'p_at_1': 0., 'recall_at_3': .5, 'mrr_at_6': .5, 'all_gold_at_3': 0.}
    assert ranking_metrics(['A@1'], {'A@2'})['recall_at_3'] == 0
    assert ranking_metrics([], set()) is None


def test_grounding_does_not_reward_empty_answers_or_hide_failed_judges():
    result = grounding_metrics([
        {'status': 'ok', 'claims': [{'verdict': v} for v in ('supported', 'unsupported', 'contradicted', 'uncertain')]},
        {'status': 'ok', 'claims': []}, {'status': 'error'}])
    assert result['faithfulness_micro'] == .25
    assert result['hallucination_micro'] == .5
    assert result['uncertain_ratio'] == .25
    assert result['failed_evaluations'] == 1
    assert grounding_metrics([{'status': 'ok', 'claims': []}])['faithfulness_micro'] is None


def test_judge_cannot_invent_answer_spans_or_source_quotes():
    good = {'abstained': False, 'claims': [{'span': '不能独立批准', 'verdict': 'supported',
        'evidence': [{'source': 1, 'quote': '不能独立批准'}]}]}
    assert validate_judgment(good, '试用期不能独立批准。[1]', [{'text': '员工不能独立批准折扣。'}]) == good
    with pytest.raises(ValueError):
        validate_judgment(good, '可以批准', [{'text': '员工不能独立批准折扣。'}])
    with pytest.raises(ValueError):
        validate_judgment(good, '不能独立批准', [{'text': '无此规定'}])


def test_version_metadata_can_ground_date_claims():
    judgment = {'abstained': False, 'claims': [{'span':'7月起生效', 'verdict':'supported',
        'evidence':[{'source':1,'quote':'effective_from: 2026-07-01'}]}]}
    assert validate_judgment(judgment, '新规7月起生效。',
                            [{'text':'新版条款', 'effective_from':'2026-07-01'}]) == judgment


def test_refusal_is_not_a_factual_hallucination_but_policy_absence_is_checkable():
    result = validate_judgment({'abstained':True, 'claims':[
        {'span':'不能确认。','verdict':'unsupported'},
        {'span':'公司没有统一额度。','verdict':'unsupported'}]},
        '不能确认。公司没有统一额度。', [])
    assert [c['span'] for c in result['claims']] == ['公司没有统一额度。']
    assert result['excluded_nonfactual_claims'][0]['span'] == '不能确认。'


def test_claim_alignment_only_repairs_terminal_punctuation():
    judgment = {'abstained':False,'claims':[{'span':'未批准不得发布折扣[3]。','verdict':'unsupported'}]}
    result = validate_judgment(judgment,'未批准不得发布折扣[3]，需核对。',[])
    assert result['claims'][0]['span'] == '未批准不得发布折扣[3]'
    assert result['claims'][0]['judge_original_span'] == '未批准不得发布折扣[3]。'
    assert judgment['claims'][0]['span'].endswith('。')
    with pytest.raises(ValueError):
        validate_judgment(judgment,'未经批准可以发布折扣[3]，需核对。',[])
