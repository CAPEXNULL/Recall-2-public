"""Controlled English v2 language rules, independent of actions and scenarios."""
import json
from pathlib import Path
import re

ADAPTER_CONFIG = json.loads(Path(__file__).with_name('semantic_adapter_config.json').read_text(encoding='utf-8'))


def alternatives(items):
    return '(?:' + '|'.join(re.escape(s) for s in sorted(items, key=len, reverse=True)) + ')'


def unsupported(clause):
    markers = alternatives(ADAPTER_CONFIG['unsupported_markers'])
    return bool(re.search(r'\b' + markers + r'\b', clause, re.I) or
                re.search(r'["“”]|\b(?:and|or)\b.*\b(?:has|is|runs|operates)\b', clause, re.I))


def number_value(raw):
    match = re.fullmatch(r'([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(.*?)', raw.strip())
    if not match:
        return None
    unit = match[2] or None
    unit = ADAPTER_CONFIG['unit_aliases'].get(unit.casefold(), unit) if unit else None
    return float(match[1]), unit


def indirect_fact(clause):
    """Explicit usability qualification is required; possession alone is insufficient."""
    nouns = alternatives(ADAPTER_CONFIG['artifact_nouns'])
    adjectives = alternatives(ADAPTER_CONFIG['usability_adjectives'])
    patterns = [
        (r'(.+?)\s+has\s+(?:a|an)\s+' + adjectives + r'\s+' + nouns, True),
        (r'(.+?)\s+has\s+no\s+' + adjectives + r'\s+' + nouns, False),
        (r'(.+?)\s+does\s+not\s+have\s+(?:a|an)\s+' + adjectives + r'\s+' + nouns, False),
        (r"(.+?)[’']s\s+" + nouns + r'\s+is\s+' + adjectives, True),
        (r"(.+?)[’']s\s+" + nouns + r'\s+is\s+not\s+' + adjectives, False),
        (nouns + r'\s+for\s+(.+?)\s+is\s+' + adjectives, True),
        (nouns + r'\s+for\s+(.+?)\s+is\s+not\s+' + adjectives, False),
    ]
    for pattern, value in patterns:
        match = re.fullmatch(pattern, clause, re.I)
        if match:
            subject = match[1].strip().casefold()
            if subject in ADAPTER_CONFIG['unresolved_subjects']:
                return None
            return {'subject': subject, 'relation': 'available', 'value': value,
                    'value_type': 'boolean', 'evidence': clause}
    phrase = alternatives(ADAPTER_CONFIG['temperature_phrases'])
    match = re.fullmatch(r'(.+?)\s+' + phrase + r'\s+(.+)', clause, re.I)
    if match and match[1].strip().casefold() not in ADAPTER_CONFIG['unresolved_subjects']:
        number = number_value(match[2])
        if number:
            value, unit = number
            result = {'subject': match[1].strip().casefold(), 'relation': 'temperature',
                      'value': value, 'value_type': 'number', 'evidence': clause}
            if unit:
                result['unit'] = unit
            return result
    return None
