"""Conservative source attestation. Unknown syntax is not a factual value.

Accepted inputs are typed source records and a closed declarative grammar.
Registry identifiers/aliases bind fields; this is not a free-text semantic oracle.
"""
import json
import re
from decimal import Decimal

MODAL = re.compile(r'\b(?:may|might|will|would|could|predicts?|predicted|forecast\w*|plans?|planned|'
                   r'hypothetical|suppose\w*|assume\w*|future|previously|formerly|was|were)\b', re.I)
REPORT = re.compile(r'^(?:a|the) (?:[\w-]+ ){0,5}(?:record|source|inspection|diagram) '
                    r'(?:explicitly )?(?:states|reports|confirms|says) (?:that )?', re.I)


def numeric_identity(value):
    """Exact numeric identity: no tolerance, no bool/number coercion."""
    exact = format(Decimal(str(value)), 'f')
    if Decimal(str(value)) == 0:return '0'
    return exact.rstrip('0').rstrip('.') if '.' in exact else exact


def _literal(value):
    return r'"?' + re.escape(value) + r'"?'


def attest(h, registry):
    span = h['provenance']['source_span']
    common = dict(subject_text=h['subject_raw'], relation_id=h['relation'],
                  span_id=span['span_id'], polarity=True, assertion_mode='ACTUAL')
    if span.get('source_kind') == 'data':
        datum = json.loads(span['exact_text'])
        if not isinstance(datum, dict) or datum.get('value', datum.get('object')) is None:
            return []
        if (datum.get('assertion_mode', 'ACTUAL') != 'ACTUAL' or
            datum.get('source_state', 'CURRENT') != 'CURRENT' or datum.get('reported', True) is not True or
            datum.get('polarity', True) is not True or
            datum.get('temporal_scope', 'current') != 'current'):
            return []
        if datum.get('subject') != h['subject_raw'] or datum.get('relation') != h['relation']:
            return []
        return [dict(common, value=datum.get('value', datum.get('object')), unit=datum.get('unit'))]

    subject = _literal(h['subject_raw'])
    types = {r['entity_kind'] for r in registry.config['roles']}
    types.update({'entity', 'node', 'object', 'device', 'instrument'})
    typed_subject = r'(?:the )?(?:(?:' + '|'.join(re.escape(x) for x in sorted(types)) + ') )?' + subject
    definition = h['definition']
    labels = [h['relation']] + definition.get('source_labels', [])
    field = '(?:' + '|'.join(re.escape(x) for x in labels) + ')'
    kind = h['value_type']
    out = []
    # Split only explicit statement boundaries; inherited modal frames cannot
    # be turned into current facts by a later fragment of the same sentence.
    for sentence in re.split(r'(?<=[.!?])\s+|\n', span['exact_text']):
        if MODAL.search(sentence):
            continue
        for clause in sentence.split(';'):
            clause = clause.strip().rstrip('.').strip()
            clause = REPORT.sub('', clause)
            clause = re.sub(r'^the source explicitly states ', '', clause, flags=re.I)
            value = unit = None
            if h['relation'] == 'role':
                role = re.escape(h['value_or_object'])
                entity_kind = re.escape(definition.get('entity_kind', ''))
                if re.fullmatch(rf'(?:the )?(?:{entity_kind} )?{subject} is (?:also )?(?:the )?{role}', clause, re.I):
                    value = h['value_or_object']
            elif kind == 'boolean':
                patterns = [rf'{subject} has {field} equal to (?P<v>true|false)',
                    rf'(?:the )?{field} value (?:for|of) {subject} is (?P<v>true|false)',
                    rf'{field}\s*=\s*(?P<v>true|false) for {subject}',
                    rf'{subject}\s*[:.]\s*{field}\s*=\s*(?P<v>true|false)']
                match = next((m for p in patterns if (m := re.fullmatch(p, clause, re.I))), None)
                if match:
                    value = match['v'].lower() == 'true'
                else:
                    # Compound field names admit a documented noun/predicate
                    # shorthand, not synonyms, antonyms or implied readiness.
                    words = h['relation'].split('_')
                    if len(words) == 2:
                        noun, predicate = map(re.escape, words)
                        m = re.fullmatch(rf'(?:the )?{noun} of {subject} is (?P<neg>not )?{predicate}', clause, re.I)
                        if m:
                            value = not bool(m['neg'])
                        elif re.fullmatch(rf'{subject} has (?:a |an )?{predicate} {noun}', clause, re.I):
                            value = True
                    elif len(words) == 1:
                        m = re.fullmatch(rf'{subject} is (?P<neg>not )?{field}', clause, re.I)
                        if m:
                            value = not bool(m['neg'])
            elif kind in {'number', 'datetime'}:
                m = re.fullmatch(rf'(?:the )?(?:(?:measured|latest|current) )?{field} of {subject} '
                    rf'(?:is|as|equals|is written as) (?P<v>.+)', clause, re.I)
                if m:
                    if kind == 'datetime':
                        value = m['v']
                    else:
                        n = re.fullmatch(r'(?P<n>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?: (?P<u>.+))?', m['v'])
                        if n:
                            d = Decimal(n['n']);value = int(d) if d == d.to_integral_value() else float(d)
                            unit = n['u']
                            if unit and definition.get('unit') is None:
                                meaning = definition['meaning'].casefold()
                                if unit.casefold() in meaning and ('count' in meaning or 'number of' in meaning):
                                    unit = None
            elif kind == 'set/list':
                m = re.fullmatch(rf'(?:the )?complete {field} set (?:for|of) {subject} is \[(?P<v>[^\[\]]*)\]', clause, re.I)
                if m:
                    value = [x.strip().strip('"') for x in m['v'].split(',') if x.strip()]
                else:
                    m = re.fullmatch(rf'for {subject}, (?P<v>.+) are the complete {field}', clause, re.I)
                    if m:
                        value = [x.strip().strip('"') for x in re.split(r',| and ', m['v'])]
            elif h['semantic_class'] == 'RELATION':
                m = re.fullmatch(rf'{typed_subject} is {field} (?P<v>.+)', clause, re.I)
                if m:
                    value = m['v'].strip('"')
            if value is not None:
                out.append(dict(common, value=value, unit=unit))
    return out
