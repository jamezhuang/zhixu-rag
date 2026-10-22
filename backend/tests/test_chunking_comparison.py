import pytest

from evaluation.chunking import body_and_sections, fixed_spans, recursive_spans, sentence_spans, semantic_spans
from scripts.compare_chunking import normalize, supplement_metrics


@pytest.mark.parametrize('text', ['短句。','第一段。\n\n第二段。第三句。\n结束。','长'*130,'a b c d e f\nmore text'])
@pytest.mark.parametrize('splitter',[fixed_spans,recursive_spans])
def test_splitters_preserve_every_character_and_bound_lengths(text,splitter):
    spans=splitter(text,9)
    assert ''.join(text[a:b] for a,b in spans)==text
    assert all(0<b-a<=9 for a,b in spans)
    assert all(left[1]==right[0] for left,right in zip(spans,spans[1:]))


def test_recursive_prefers_sentence_boundary_while_fixed_can_break_condition():
    text='准备好材料。未经主管批准不得确认折扣。'
    fixed=[text[a:b] for a,b in fixed_spans(text,14)]
    recursive=[text[a:b] for a,b in recursive_spans(text,14)]
    condition='未经主管批准不得确认折扣。'
    assert not any(condition in c for c in fixed)
    assert any(condition in c for c in recursive)


def test_heading_metadata_is_available_to_all_alternatives():
    text,sections=body_and_sections('# 制度\n说明。\n## 限制\n不能批准。\n')
    assert '#' not in text
    assert any(name=='限制' for _,_,name in sections)
    assert '不能批准。' in text
    crlf,_=body_and_sections('# 制度\r\n说明。\r\n## 限制\r\n不能批准。\r\n')
    assert crlf==text


def test_semantic_breaks_on_low_similarity_and_preserves_body():
    text='活动审批说明。'*10+'指标计算说明。'*10
    units=sentence_spans(text)
    similarities=[.9]*(len(units)-1)
    similarities[9]=.1
    spans=semantic_spans(text,units,similarities,.5,2200)
    assert len(spans)==2
    assert normalize(''.join(text[a:b] for a,b in spans))==normalize(text)
    assert spans[0][1]==len('活动审批说明。'*10)


def test_quote_metric_requires_correct_version_and_single_complete_chunk():
    case={'answerable':True,'gold':[{'document':'A@2','quote':'必须先审核再执行'}]}
    rows=[{'document_id':'A@2','text':'必须先审核'},{'document_id':'A@2','text':'再执行'}]
    report={'chunking_config':{'cap':10},'results':[{'case':case,'retrieved':{'hybrid_rerank':rows}}],
            'retrieval_summary':{'hybrid_rerank':{}}}
    result=supplement_metrics(report,rows,[{'key':'A@2','body':'必须先审核再执行。'}])
    assert result['quote_recall_at_6']==0
    assert result['protected_integrity']==0


def test_semantic_length_fallback_preserves_sentences_when_they_fit():
    text='准备好材料。未经主管批准不得确认折扣。'
    spans=semantic_spans(text,sentence_spans(text),[.9],.1,14)
    assert any('未经主管批准不得确认折扣。' in text[a:b] for a,b in spans)
    assert all(b-a<=14 for a,b in spans)


def test_report_save_retries_transient_windows_file_lock(tmp_path,monkeypatch):
    from pathlib import Path
    from scripts.benchmark_retrieval import save
    original=Path.replace
    calls=[]
    def locked(self,target):
        calls.append(1)
        if len(calls)<3:
            raise PermissionError('temporary lock')
        return original(self,target)
    monkeypatch.setattr(Path,'replace',locked)
    monkeypatch.setattr('scripts.benchmark_retrieval.time.sleep',lambda seconds:None)
    target=tmp_path/'result.json'
    save({'ok':True},target)
    assert len(calls)==3 and 'true' in target.read_text()
