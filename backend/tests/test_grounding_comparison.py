import pytest

from scripts.benchmark_retrieval import METHODS, evaluate_answer


@pytest.mark.parametrize('method', METHODS)
def test_grounding_compares_requested_method_without_using_gold(method):
    row = {'case':{'question':'测试问题','as_of':'2026-10-21','gold':[{'secret':'must not enter model'}]},
           'retrieved':{m:[{'text':'证据-' + m}] for m in METHODS}}
    class Fake:
        def json(self, prompt, data):
            assert 'gold' not in data
            assert data['evidence'][0]['text'].split('\n\n')[0] == '证据-' + method
            if prompt.startswith('你是'):
                return {'answer':'证据-' + method}
            return {'abstained':False, 'claims':[{'span':'证据-' + method,'verdict':'supported',
                    'evidence':[{'source':1,'quote':'证据-' + method}]}]}
    result = evaluate_answer(row,Fake(),method)
    assert result['status'] == 'ok'
    assert result['answer'] == '证据-' + method


def test_failed_judgment_resume_keeps_original_answer():
    row = {'case':{'question':'测试问题','as_of':'2026-10-21'},
           'retrieved':{'bm25':[{'text':'原始证据'}]},
           'grounding':{'status':'error','answer':'原有回答'}}
    class Fake:
        def json(self,prompt,data):
            assert not prompt.startswith('你是')
            assert data['answer'] == '原有回答'
            return {'abstained':False, 'claims':[{'span':'原有回答','verdict':'unsupported','evidence':[]}]}
    result = evaluate_answer(row,Fake(),'bm25')
    assert result['status'] == 'ok'
    assert result['answer'] == '原有回答'
