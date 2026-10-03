"""Conservative applicability contract; bounded English, not general NLP."""
from copy import deepcopy
import math
import re

STATES = {'CONFIRMED', 'VIOLATED', 'UNKNOWN'}
CATEGORIES = {'Availability', 'Permission', 'Compatibility', 'Causality',
              'Freshness', 'Provenance / Scope', 'Validity / Trustworthiness', 'Completeness'}


def validate_preconditions(value):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError('Preconditions must be a list of at most 32 declarations.')
    ids = set()
    for p in value:
        if isinstance(p, dict) and 'operator' in p:
            required = {'id', 'category', 'subject', 'relation', 'operator', 'mandatory', 'expected_value_type'}
            if (not required <= set(p) or ('expected_value' in p) == ('expected_object' in p) or
                    not isinstance(p['id'], str) or not p['id'] or p['id'] in ids or
                    type(p['mandatory']) is not bool or not isinstance(p['relation'], str) or not p['relation'] or
                    not isinstance(p['operator'], str) or not isinstance(p['expected_value_type'], str)):
                raise ValueError('Invalid typed precondition declaration.')
            ids.add(p['id'])
            continue
        if (not isinstance(p, dict) or set(p) != {'id', 'category', 'requirement', 'mandatory', 'expected'} or
                not isinstance(p['id'], str) or not p['id'].strip() or p['id'] in ids or
                not isinstance(p['category'], str) or not p['category'].strip() or
                type(p['mandatory']) is not bool or
                not isinstance(p['requirement'], (dict, str)) or not p['requirement'] or
                type(p['expected']) not in (bool, str, int, float) or
                isinstance(p['expected'], float) and not math.isfinite(p['expected'])):
            raise ValueError('Invalid precondition declaration.')
        ids.add(p['id'])
    return deepcopy(value)


def unknown(reason):
    return {'evaluation': 'UNKNOWN', 'evidence': 'No unambiguous supporting evidence.', 'reason': reason}


def aggregate(evaluated):
    """No similarity, entity, or category-specific rules."""
    mandatory = [row['evaluation'] for row in evaluated if row['precondition']['mandatory']]
    if 'VIOLATED' in mandatory:
        return 'VIOLATED'
    if not mandatory or 'UNKNOWN' in mandatory:
        return 'UNKNOWN'
    return 'CONFIRMED'


def check_applicability(situation, preconditions, evaluator, extra_checks=None):
    conditions = validate_preconditions(preconditions)
    evaluated = []
    for p in conditions:
        try:
            result = evaluator.evaluate(deepcopy(situation), deepcopy(p))
            if (not isinstance(result, dict) or set(result) != {'evaluation', 'evidence', 'reason'} or
                    not isinstance(result['evaluation'], str) or result['evaluation'] not in STATES or
                    not isinstance(result['evidence'], str) or not result['evidence'].strip() or
                    not isinstance(result['reason'], str) or not result['reason'].strip()):
                result = unknown('Malformed evaluator result or missing evidence/reason.')
        except Exception as exc:
            result = unknown('Evaluator failure: ' + type(exc).__name__)
        evaluated.append({'precondition': p, **result})
    if not any(p['mandatory'] for p in conditions):
        evaluated.append({'precondition': {'id': 'experience:mandatory-preconditions', 'category': 'Completeness', 'mandatory': True},
                          **unknown('No mandatory Experience preconditions declared.')})
    evaluated.extend(deepcopy(extra_checks or []))
    state = aggregate(evaluated)
    reasons = [r['reason'] for r in evaluated if r['precondition']['mandatory'] and r['evaluation'] != 'CONFIRMED']
    return {'evaluated_preconditions': evaluated, 'applicability': state,
            'decision': 'TRANSFER' if state == 'CONFIRMED' else 'REJECT',
            'reason': ' '.join(reasons) if reasons else ('All mandatory preconditions confirmed.' if
                      state == 'CONFIRMED' else 'No mandatory preconditions declared; applicability is UNKNOWN.'),
            'evaluator_calls': len(conditions), 'evaluator_model_calls': 0}


def aliases(values):
    if not isinstance(values, list) or not values or any(not isinstance(s, str) or not s.strip() for s in values):
        raise ValueError('Nonempty entity aliases required.')
    return r'(?<!\w)(?:' + '|'.join(re.escape(s.casefold()) for s in sorted(values, key=len, reverse=True)) + r')(?!\w)'


class LocalEvaluator:
    """Structured facts match full category/requirement, hence object/scope.

    Facts are attributed input, not independently verified reality. Text supports
    four categories in a controlled assertion grammar; other inputs stay UNKNOWN.
    """
    def evaluate(self, situation, precondition):
        category, req = precondition['category'], precondition['requirement']
        if category not in CATEGORIES:
            return unknown('Unsupported category: ' + category)
        values, evidence = [], []
        if isinstance(situation, dict):
            facts = situation.get('facts', [])
            if not isinstance(facts, list):
                return unknown('Malformed structured facts.')
            for fact in facts:
                if isinstance(fact, dict) and fact.get('category') == category and fact.get('requirement') == req:
                    if (set(fact) != {'category', 'requirement', 'value', 'evidence'} or
                            type(fact['value']) not in (bool, str, int, float) or
                            isinstance(fact['value'], float) and not math.isfinite(fact['value']) or
                            not isinstance(fact['evidence'], str) or not fact['evidence'].strip()):
                        return unknown('Malformed structured assertion or missing evidence.')
                    values.append(fact['value'])
                    evidence.append(fact['evidence'])
            text = situation.get('text', '')
        else:
            text = situation
        if not isinstance(text, str):
            return unknown('Unsupported situation format.')
        if category in {'Availability', 'Permission', 'Compatibility', 'Causality'} and text:
            assertions = self.text_assertions(text, category, req)
            if assertions is None:
                return unknown('Unsupported or ambiguous condition/interpretation.')
            for value, quote in assertions:
                values.append(value)
                evidence.append(quote)
        if not values:
            return unknown('Insufficient evidence for this condition/object/scope.')
        if any(type(v) is not type(values[0]) or v != values[0] for v in values):
            return unknown('Contradictory evidence for this condition/object/scope.')
        expected = precondition['expected']
        matches = type(values[0]) is type(expected) and values[0] == expected
        return {'evaluation': 'CONFIRMED' if matches else 'VIOLATED', 'evidence': '\n'.join(evidence),
                'reason': 'Explicit attributed assertion matches requirement.' if matches else
                          'Explicit attributed assertion violates requirement.'}

    def text_assertions(self, text, category, req):
        if not isinstance(req, dict) or set(req) - {'subject', 'object', 'resources'} or 'subject' not in req:
            return None
        try:
            subject = aliases(req['subject'])
            obj = aliases(req['object']) if req.get('object') else None
            resources = [aliases(item) for item in req.get('resources', [])]
        except (TypeError, ValueError):
            return None
        clauses = [s.strip() for s in re.split(r'[.;!?\n]+', text) if s.strip()]
        results = []
        for i, quote in enumerate(clauses):
            clause = quote.casefold()
            refers = re.search(subject, clause) or (obj and re.search(obj, clause)) or any(re.search(r, clause) for r in resources)
            if refers and (re.search(r'\b(?:if|unless|might|maybe|possibly|probably|reportedly)\b', clause) or
                           re.search(r'\b(?:may|could)\s+be\b', clause)):
                return None
            value = None
            if category in {'Availability', 'Permission'}:
                prefix = rf'^(?:the\s+)?{subject}(?:\s+it)?\s+(?:is|are|remains|remain)\s+'
                if category == 'Availability':
                    positive = r'(?:available|accessible|enabled|possible)'
                    negative = r'(?:unavailable|inaccessible|disabled|impossible|not\s+(?:available|accessible|enabled|possible))'
                else:
                    positive = r'(?:permitted|allowed|authorized|not\s+(?:forbidden|prohibited|disallowed))'
                    negative = r'(?:forbidden|prohibited|disallowed|not\s+(?:permitted|allowed|authorized))'
                # Unparsed scope/qualifiers cannot silently confirm a prefix.
                tail = r'(?:\s+with\s+(?:a\s+)?(?:short|brief|\d+)\s+delay)?' if category == 'Availability' else ''
                if re.fullmatch(prefix + negative + tail, clause):
                    value = False
                elif re.fullmatch(prefix + positive + tail, clause):
                    value = True
                elif re.search(prefix + rf'(?:{positive}|{negative})\b', clause):
                    return None
                if value is not None and re.search(r'\bit\b', clause):
                    antecedent = clauses[i - 1].casefold() if i else ''
                    if not obj or not re.search(obj, antecedent):
                        return None
                if category == 'Availability':
                    for resource in resources:
                        removed = re.search(rf'^(?:the\s+)?{resource}\s+(?:(?:has|have)\s+been|is|was)\s+(?:deleted|removed|destroyed)\b', clause)
                        following = clauses[i + 1].casefold() if i + 1 < len(clauses) else ''
                        if removed and re.fullmatch(r'no\s+(?:replacement|backup|alternative)\s+(?:exists|is available)', following):
                            value = False
                            quote += '; ' + clauses[i + 1]
            elif category == 'Compatibility' and obj:
                prefix = rf'^(?:the\s+)?{subject}\s+(?:is|are|remains|remain)\s+'
                if re.fullmatch(prefix + rf'(?:incompatible|not compatible)\s+with\s+(?:the\s+)?{obj}', clause):
                    value = False
                elif re.fullmatch(prefix + rf'compatible\s+(?:with\s+(?:the\s+)?{obj}|after\s+(?:the\s+)?{obj}\s+migration)', clause):
                    value = True
                else:
                    reverse = rf'^(?:the\s+)?{obj}\s+'
                    suffix = rf'\s+be\s+read\s+by\s+(?:the\s+)?{subject}'
                    if re.fullmatch(reverse + r'(?:cannot|can not)' + suffix, clause):
                        value = False
                    elif re.fullmatch(reverse + 'can' + suffix, clause):
                        value = True
                    elif re.search(prefix + r'(?:compatible|incompatible|not compatible)\b', clause):
                        return None
            elif category == 'Causality' and obj:
                prefix = rf'^(?:the\s+)?{subject}\s+(?:started|began|occurred)\s+(?:immediately\s+)?'
                if re.fullmatch(prefix + rf'before\s+(?:the\s+)?{obj}', clause):
                    value = False
                elif re.fullmatch(prefix + rf'after\s+(?:the\s+)?{obj}', clause):
                    value = True
                elif re.search(prefix + r'(?:after|before)\b', clause):
                    return None
            if value is not None:
                results.append((value, quote))
        return results
