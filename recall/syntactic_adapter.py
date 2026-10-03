"""Domain-independent structural slots resolved through a separate semantic lexicon."""
import json
from pathlib import Path
import re
from semantic_adapter import number_value

LEXICON = json.loads(Path(__file__).with_name('syntactic_lexicon.json').read_text(encoding='utf-8'))


def target(value):
    value = value.strip().casefold()
    if (not value or value in LEXICON['unresolved_targets'] or
            re.search(r'[,;"“”]|\b(?:and|or|either|both|for)\b', value) or
            not re.fullmatch(r'[\w-]+(?:\s+[\w-]+)*', value)):
        return None
    return value


def nominal(value):
    words = value.strip().casefold().split()
    absent = words and words[0] == 'no'
    if words and words[0] in {'a', 'an', 'the', 'no'}:
        words = words[1:]
    if not words or any(not re.fullmatch(r'[a-z-]+', w) for w in words):
        return None
    return {'head': words[-1], 'modifiers': words[:-1], 'absent': bool(absent)}


def structure(clause):
    # Order rules contain grammatical slots only, never semantic nouns/states.
    patterns = [
        ('predicate', r'For (?P<target>[^,]+),\s*(?P<nominal>.+?) is (?P<predicate>(?:not )?[\w-]+)'),
        ('existence', r'There is (?P<nominal>.+?) for (?P<target>.+)'),
        ('predicate', r'(?P<nominal>(?:a|an|the|no) .+?) is (?P<predicate>(?:not )?[\w-]+) for (?P<target>.+)'),
        ('predicate', r'(?P<nominal>.+?) for (?P<target>.+?) is (?P<predicate>(?:not )?[\w-]+)'),
        ('predicate', r"(?P<target>.+?)[’']s (?P<nominal>.+?) is (?P<predicate>(?:not )?[\w-]+)"),
        ('absence', r'(?P<target>.+?) does not have (?P<nominal>.+)'),
        ('existence', r'(?P<target>.+?) has (?P<nominal>.+)'),
    ]
    for kind, pattern in patterns:
        match = re.fullmatch(pattern, clause, re.I)
        if match:
            slots = match.groupdict()
            slots['kind'] = kind
            return slots
    return None


def semantic_fact(slots, evidence):
    subject, n = target(slots['target']), nominal(slots['nominal'])
    if not subject or not n:
        return []
    cls = LEXICON['nominal_heads'].get(n['head'])
    if cls is None:
        return []
    modifiers = [LEXICON['states'].get(word) for word in n['modifiers']]
    if any(m is None for m in modifiers):
        return []
    predicate = slots.get('predicate')
    negated = bool(predicate and predicate.casefold().startswith('not '))
    if predicate:
        state = LEXICON['states'].get(predicate.casefold().removeprefix('not '))
        if state is None:
            return []
        relation, value = state['relation'], state['value'] != negated
    elif modifiers and len({(m['relation'], m['value']) for m in modifiers}) == 1:
        relation, value = modifiers[0]['relation'], modifiers[0]['value']
    else:
        return []
    if relation not in LEXICON['classes'][cls] or any(m['relation'] != relation for m in modifiers):
        return []
    if len({m['value'] for m in modifiers}) > 1:
        return []
    absent = n['absent'] or slots['kind'] == 'absence'
    if absent:
        # Absence of an unavailable object cannot establish availability.
        if not value or any(not m['value'] for m in modifiers):
            return []
        value = False
    return [{'subject': subject, 'relation': relation, 'value': value,
             'value_type': 'boolean', 'evidence': evidence}]


def numeric_structure(clause):
    patterns = [
        r'For (?P<target>[^,]+),\s*(?:the )?(?P<property>[\w-]+) is (?P<value>.+)',
        r'(?:the )?(?P<property>[\w-]+) for (?P<target>.+?) is (?P<value>.+)',
        r'There is a (?P<property>[\w-]+) of (?P<value>.+?) for (?P<target>.+)',
        r'(?P<value>.+?) is the (?P<property>[\w-]+) for (?P<target>.+)',
    ]
    for pattern in patterns:
        match = re.fullmatch(pattern, clause, re.I)
        if match:
            slots = match.groupdict()
            definition = LEXICON['numeric_properties'].get(slots['property'].casefold())
            # A structurally numeric statement with unknown property is not reinterpreted.
            if definition is None:
                return [] if number_value(slots['value']) is not None else None
            subject, number = target(slots['target']), number_value(slots['value'])
            if not subject or number is None:
                return []
            value, unit = number
            fact = {'subject': subject, 'relation': definition['relation'], 'value': value,
                    'value_type': definition['value_type'], 'evidence': clause}
            if unit:
                fact['unit'] = unit
            return [fact]
    return None


def syntactic_facts(clause):
    numeric = numeric_structure(clause)
    if numeric is not None:
        return numeric
    slots = structure(clause)
    if slots is not None and nominal(slots['nominal']) is None:
        return None
    return semantic_fact(slots, clause) if slots is not None else None
