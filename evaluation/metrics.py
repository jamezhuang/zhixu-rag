"""Explicit document-version ranking and claim-level grounding metrics."""
import re


def ranking_metrics(ranked_documents, relevant):
    ranked = list(dict.fromkeys(ranked_documents))[:6]
    gold = set(relevant)
    if not gold:
        return None  # Negative controls are not assigned an artificial recall.
    return {'p_at_1': float(bool(ranked) and ranked[0] in gold),
            'recall_at_3': len(set(ranked[:3]) & gold) / len(gold),
            'mrr_at_6': next((1 / (i + 1) for i, key in enumerate(ranked) if key in gold), 0.0),
            'all_gold_at_3': float(gold <= set(ranked[:3]))}


def grounding_metrics(evaluations):
    valid = [e for e in evaluations if e.get('status') == 'ok']
    claims = [c for e in valid for c in e['claims']]
    total = len(claims)
    bad = {'unsupported', 'contradicted'}
    substantive = [e for e in valid if e['claims']]
    count = lambda verdict: sum(c['verdict'] == verdict for c in claims)
    return {'evaluated_answers': len(valid), 'failed_evaluations': len(evaluations) - len(valid),
            'answers_without_checkable_claims': sum(not e['claims'] for e in valid),
            'excluded_nonfactual_spans': sum(len(e.get('excluded_nonfactual_claims', [])) for e in valid),
            'checkable_claims': total, 'supported_claims': count('supported'),
            'unsupported_claims': count('unsupported'), 'contradicted_claims': count('contradicted'),
            'uncertain_claims': count('uncertain'),
            'faithfulness_micro': count('supported') / total if total else None,
            'hallucination_micro': sum(c['verdict'] in bad for c in claims) / total if total else None,
            'uncertain_ratio': count('uncertain') / total if total else None,
            'answer_hallucination_rate': sum(any(c['verdict'] in bad for c in e['claims']) for e in substantive) / len(substantive) if substantive else None}


def source_text(item):
    return item['text'] + '\n\n' + '\n'.join(
        f'{key}: {item[key]}' for key in ('code', 'version', 'effective_from', 'effective_to') if key in item)


def validate_judgment(result, answer, evidence):
    """Reject invented answer spans and fabricated support quotes, not semantic judge errors."""
    if not isinstance(result.get('claims'), list) or not isinstance(result.get('abstained'), bool):
        raise ValueError('Invalid grounding judgment schema')
    sources = {str(i + 1): source_text(x) for i, x in enumerate(evidence)}
    seen = set()
    included, excluded = [], []
    for claim in result['claims']:
        span, verdict = claim.get('span'), claim.get('verdict')
        if isinstance(span, str) and span not in answer:
            aligned = span.strip().rstrip('。.!！?？；;，,')
            if aligned and aligned in answer:
                claim = {**claim, 'span':aligned, 'judge_original_span':span,
                         'span_alignment':'terminal_punctuation_only'}
                span = aligned
        if not isinstance(span, str) or not span.strip() or span not in answer or span in seen:
            raise ValueError('Judge claim is not a unique exact answer span: ' + str(span)[:200])
        seen.add(span)
        if verdict not in ('supported', 'unsupported', 'contradicted', 'uncertain'):
            raise ValueError('Invalid claim verdict')
        # A model sometimes labels its own inability to confirm as a factual
        # hallucination despite the rubric. Exclude only standalone disclaimers;
        # statements such as "公司没有统一额度" remain checkable claims.
        if re.fullmatch(r'(?:无法|不能)(?:进一步)?确认[^。！？；\n]{0,40}[。.]?', span.strip()) or re.fullmatch(
                r'未提供[^。！？；\n]{0,30}[，,](?:不能|无法)进一步确认[。.]?', span.strip()):
            excluded.append(claim)
            continue
        if verdict in ('supported', 'contradicted'):
            supports = claim.get('evidence', [])
            if not supports:
                raise ValueError('Supported/contradicted claim needs source evidence')
            for support in supports:
                quote = support.get('quote')
                if not isinstance(quote, str) or not quote.strip() or quote not in sources.get(str(support.get('source')), ''):
                    raise ValueError('Judge support quote is not in retrieved evidence')
        included.append(claim)
    return {**result, 'claims':included, **({'excluded_nonfactual_claims':excluded} if excluded else {})}
