"""Explicit evaluation splitters; production heading splitter is reused separately."""
import re


def body_and_sections(raw):
    raw = raw.replace('\r\n','\n').replace('\r','\n')
    text, sections, current = '', [], '正文'
    for line in raw.splitlines(keepends=True):
        if re.match(r'^#{1,6}\s+', line):
            current = re.sub(r'^#{1,6}\s+', '', line).strip()
        else:
            sections.append((len(text), len(text)+len(line), current))
            text += line
    return text, sections


def fixed_spans(text, cap):
    return [(i,min(i+cap,len(text))) for i in range(0,len(text),cap)]


def recursive_spans(text, cap):
    """Paragraph -> sentence -> character fallback, without overlap or lost chars."""
    def split(start, end, level):
        if end-start <= cap:
            return [(start,end)]
        separators = [r'\n\s*\n',r'[。！？；]\s*',r'\n',r'\s+']
        if level >= len(separators):
            return [(start+i,min(start+i+cap,end)) for i in range(0,end-start,cap)]
        cuts = [start] + [start+m.end() for m in re.finditer(separators[level],text[start:end])] + [end]
        cuts = sorted(set(cuts))
        if len(cuts) <= 2:
            return split(start,end,level+1)
        result, left, right = [], start, start
        for a,b in zip(cuts,cuts[1:]):
            if b-a > cap:
                if right>left:
                    result.append((left,right))
                result.extend(split(a,b,level+1))
                left=right=b
            elif b-left > cap:
                if right>left:
                    result.append((left,right))
                left,right=a,b
            else:
                right=b
        if right>left:
            result.append((left,right))
        return result
    return split(0,len(text),0) if text else []


def sentence_spans(text):
    cuts = [0] + [m.end() for m in re.finditer(r'[。！？；]\s*|\n\s*\n',text)] + [len(text)]
    cuts = sorted(set(cuts))
    return [(a,b) for a,b in zip(cuts,cuts[1:]) if text[a:b].strip()]


def semantic_spans(text, units, similarities, threshold, cap):
    """Split at low adjacent sentence similarity, min 64 chars, then hard cap."""
    if not units:
        return []
    cuts, left = [], units[0][0]
    for i in range(1,len(units)):
        if similarities[i-1] < threshold and units[i][0]-left >= 64:
            cuts.append((left,units[i][0]))
            left=units[i][0]
    cuts.append((left,units[-1][1]))
    return [(a+x,a+y) for a,b in cuts for x,y in recursive_spans(text[a:b],cap)]


def section_for(start,end,sections):
    names = list(dict.fromkeys(name for a,b,name in sections if a<end and b>start))
    return ' / '.join(names)[:300]
