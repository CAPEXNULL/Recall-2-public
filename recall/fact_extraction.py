"""Pluggable fact extraction with source/evidence support validation."""
from copy import deepcopy
import re
from fact_model import CONFIG, canonical, dimension
from semantic_adapter import indirect_fact, number_value, unsupported
from syntactic_adapter import syntactic_facts


def assertion(subject, relation, value, value_type, evidence, unit=None):
    result = {'subject': subject.strip().casefold(), 'relation': relation,
              'value': value, 'value_type': value_type, 'evidence': evidence}
    if unit:
        result['unit'] = unit
    return result


def parse_clause(clause):
    """Bounded compositional English grammar driven by a versioned lexicon."""
    text = clause.strip()
    lowered = text.casefold()
    if unsupported(text):
        return []
    structural = syntactic_facts(text)
    if structural is not None:
        return structural
    indirect = indirect_fact(text)
    if indirect is not None:
        return [indirect]
    for phrase in sorted(CONFIG['target_phrases'], key=len, reverse=True):
        match = re.fullmatch(re.escape(phrase) + r'\s+(.+)', lowered)
        if match:
            return [assertion(match[1], 'role', 'current_target', 'string/entity', text)]
    for relation, polarities in CONFIG['boolean_relations'].items():
        for polarity, phrases in polarities.items():
            for phrase in phrases:
                match = re.fullmatch(r'(.+?)\s+' + re.escape(phrase), lowered)
                if match:
                    return [assertion(match[1], relation, polarity == 'true', 'boolean', text)]
    for relation, definition in CONFIG['value_relations'].items():
        for phrase in definition['phrases']:
            match = re.fullmatch(r'(.+?)\s+' + re.escape(phrase) + r'\s+(.+)', text, flags=re.I)
            if not match:
                continue
            subject, raw = match[1], match[2].strip()
            kind, unit, value = definition['type'], None, raw
            if kind == 'number':
                numeric = number_value(raw)
                if not numeric:
                    return []
                value, unit = numeric
            elif kind == 'set/list':
                value = [s.strip() for s in re.split(r',|\band\b', raw.strip('{}[]'), flags=re.I) if s.strip()]
            return [assertion(subject, relation, value, kind, text, unit)]
    return []


def clauses_for(text):
    # Decimal dots stay inside numeric values. Timestamps use ISO offsets.
    return [s.strip() for s in re.split(r';|\n|(?<!\d)\.(?:\s+|$)|(?<=\d)\.(?=\s|$)', text) if s.strip()]


def text_facts(text):
    facts = []
    for clause in clauses_for(text):
        facts.extend(parse_clause(clause))
    return facts


class ControlledEnglishAdapter:
    model_calls = 0

    def extract(self, text, context=None):
        facts = text_facts(text)
        for i, f in enumerate(facts, 1):
            f.update(fact_id=f'text-{i}', source={'kind': 'text'}, confidence=None)
        return {'facts': facts, 'status': 'OK', 'errors': []}


class FactExtractor:
    def __init__(self, semantic_adapter=None):
        self.semantic_adapter = semantic_adapter if semantic_adapter is not None else ControlledEnglishAdapter()

    def extract(self, text, context=None):
        context = context or []
        accepted, rejected, errors = [], [], []
        before_calls = getattr(self.semantic_adapter, 'model_calls', 0)
        try:
            raw = self.semantic_adapter.extract(text, deepcopy(context))
            if not isinstance(raw, dict) or set(raw) != {'facts', 'status', 'errors'} or not isinstance(raw['facts'], list) or raw['status'] not in {'OK', 'ERROR'} or not isinstance(raw['errors'], list):
                raise ValueError('Malformed extraction result.')
            if raw['status'] == 'ERROR':
                errors.append('Semantic adapter reported failure.')
                claims = []
            else:
                claims = raw['facts']
        except Exception as exc:
            errors.append('Semantic adapter failure: ' + type(exc).__name__)
            claims = []
        if not isinstance(context, list):
            errors.append('Malformed structured context.')
            context = []
        for i, data in enumerate(context):
            if isinstance(data, dict):
                f = deepcopy(data)
                f.update(fact_id=f'data-{i + 1}', source={'kind': 'data', 'index': i},
                         evidence=f'/context/{i}', confidence=None)
                claims.append(f)
        seen_ids = set()
        for fact in claims:
            try:
                f = deepcopy(fact)
                required = {'fact_id', 'subject', 'relation', 'value_type', 'source', 'evidence'}
                if not isinstance(f, dict) or not required <= set(f) or ('value' in f) == ('object' in f):
                    raise ValueError('Malformed fact schema.')
                if not all(isinstance(f[k], str) and f[k].strip() for k in ['fact_id', 'subject', 'relation', 'evidence']):
                    raise ValueError('Missing identity/evidence.')
                value = f.get('value', f.get('object'))
                canonical(value, f['value_type'], f.get('unit'))
                source = f['source']
                if source == {'kind': 'text'}:
                    if f['evidence'] not in clauses_for(text):
                        raise ValueError('Evidence is not a complete input clause.')
                    support = parse_clause(f['evidence'])
                    keys = ['subject', 'relation', 'value_type']
                    if not any(all(f[k].strip().casefold() == s[k].strip().casefold() for k in keys) and
                               (f['value_type'] != 'number' or dimension(f.get('unit')) == dimension(s.get('unit'))) and
                               canonical(value, f['value_type'], f.get('unit')) == canonical(s['value'], s['value_type'], s.get('unit'))
                               for s in support):
                        raise ValueError('Evidence does not support asserted fact.')
                elif isinstance(source, dict) and source.get('kind') == 'data':
                    i = source.get('index')
                    if type(i) is not int or not 0 <= i < len(context) or f['evidence'] != f'/context/{i}':
                        raise ValueError('Invalid data evidence pointer.')
                    s = context[i]
                    if any(f.get(k) != s.get(k) for k in ['subject', 'relation', 'value', 'object', 'value_type', 'unit']):
                        raise ValueError('Structured evidence mismatch.')
                else:
                    raise ValueError('Unsupported evidence source.')
                if f['fact_id'] in seen_ids:
                    raise ValueError('Duplicate fact ID.')
                seen_ids.add(f['fact_id'])
                f['subject'], f['relation'] = f['subject'].strip().casefold(), f['relation'].strip().casefold()
                f.setdefault('confidence', None)
                accepted.append(f)
            except Exception as exc:
                rejected.append({'fact': deepcopy(fact), 'reason': str(exc) if isinstance(exc, ValueError) else type(exc).__name__})
        return {'facts': accepted, 'rejected_facts': rejected, 'status': 'ERROR' if errors else 'OK',
                'errors': errors, 'extraction_calls': 1,
                'extraction_model_calls': getattr(self.semantic_adapter, 'model_calls', 0) - before_calls}
