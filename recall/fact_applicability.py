"""Fact/role resolution and parameterized actions feeding the unchanged aggregate."""
from copy import deepcopy
from string import Formatter
from applicability import unknown
from fact_model import canonical, compare, dimension


def validate_action(action):
    if action is None:
        return None
    if (not isinstance(action, dict) or not isinstance(action.get('action_type'), str) or
            not action['action_type'].strip() or not isinstance(action.get('parameters'), list)):
        raise ValueError('Invalid structured action.')
    names = set()
    for p in action['parameters']:
        if (not isinstance(p, dict) or set(p) != {'name', 'role', 'mandatory'} or
                not isinstance(p['name'], str) or not p['name'] or p['name'] in names or
                not isinstance(p['role'], str) or not p['role'] or type(p['mandatory']) is not bool):
            raise ValueError('Invalid action parameter.')
        names.add(p['name'])
    if 'display_template' in action and not isinstance(action['display_template'], str):
        raise ValueError('Invalid action display template.')
    for _, field, spec, conversion in Formatter().parse(action.get('display_template', '')):
        if field is not None and (field not in names or spec or conversion):
            raise ValueError('Display template references undeclared action parameter.')
    return deepcopy(action)


def bindings(action, extraction):
    result, checks = [], []
    if action is None:
        checks.append({'precondition': {'id': 'binding:structured-action', 'category': 'Binding', 'mandatory': True},
                       **unknown('Legacy action lacks explicit structured configuration.')})
        return result, checks
    for parameter in action['parameters']:
        facts = [f for f in extraction['facts'] if f['relation'] == 'role' and
                 f['value_type'] == 'string/entity' and
                 canonical(f.get('value', f.get('object')), 'string/entity') == parameter['role'].casefold()]
        entities = sorted({f['subject'] for f in facts})
        status = 'BOUND' if len(entities) == 1 else ('AMBIGUOUS' if entities else 'MISSING')
        row = {'name': parameter['name'], 'role': parameter['role'], 'mandatory': parameter['mandatory'],
               'status': status, 'bound_entity': entities[0] if status == 'BOUND' else None,
               'evidence': [f['evidence'] for f in facts], 'source_fact_ids': [f['fact_id'] for f in facts]}
        result.append(row)
        checks.append({'precondition': {'id': 'binding:' + parameter['name'], 'category': 'Binding',
                                       'mandatory': parameter['mandatory']},
                       'evaluation': 'CONFIRMED' if status == 'BOUND' else 'UNKNOWN',
                       'evidence': '; '.join(row['evidence']) or 'No role evidence.',
                       'reason': 'Parameter binding: ' + status})
    return result, checks


def render_action(action, bound):
    values = {p['name']: p['bound_entity'] for p in bound if p['status'] == 'BOUND'}
    resolved = {'action_type': action['action_type'], 'parameters': values}
    # Display templates have no execution capability; unresolved fields are invalid configuration.
    template = action.get('display_template')
    display_values = {p['name']: values.get(p['name'], '?') for p in action['parameters']}
    display = template.format_map(display_values) if template else action['action_type'] + ' ' + ', '.join(f'{k}={v}' for k, v in values.items())
    return resolved, display.strip()


class FactEvaluator:
    def __init__(self, extraction, bound):
        self.extraction, self.bound = extraction, bound

    def role(self, role):
        matches = {p['bound_entity'] for p in self.bound if p['role'] == role and p['status'] == 'BOUND'}
        if len(matches) != 1 or any(p['role'] == role and p['status'] != 'BOUND' for p in self.bound):
            raise ValueError('Unresolved or ambiguous role reference.')
        return next(iter(matches))

    def subject(self, value):
        return self.role(value['role_ref']) if isinstance(value, dict) and set(value) == {'role_ref'} else canonical(value, 'string/entity')

    def find(self, subject, relation):
        subject, relation = self.subject(subject), relation.casefold()
        candidates = [f for f in self.extraction['facts'] if f['subject'] == subject and f['relation'] == relation]
        for rejection in self.extraction['rejected_facts']:
            f = rejection['fact']
            if isinstance(f, dict) and str(f.get('subject', '')).casefold() == subject and str(f.get('relation', '')).casefold() == relation:
                raise ValueError('Relevant assertion has unsupported evidence/type.')
        if not candidates:
            raise ValueError('Required fact missing or extraction unsupported.')
        keys = {(f['value_type'], dimension(f.get('unit')) if f['value_type'] == 'number' else None,
                 str(canonical(f.get('value', f.get('object')), f['value_type'], f.get('unit')))) for f in candidates}
        if len(keys) != 1:
            raise ValueError('Contradictory relevant facts.')
        return candidates[0]

    def operand(self, value, value_type, unit=None):
        if isinstance(value, dict):
            if set(value) == {'literal'}:
                return value['literal'], value_type, unit, None
            if set(value) == {'role_ref'}:
                return self.role(value['role_ref']), 'string/entity', None, None
            if set(value) == {'fact_ref'}:
                ref = value['fact_ref']
                if isinstance(ref, str):
                    matches = [f for f in self.extraction['facts'] if f['fact_id'] == ref]
                    if len(matches) != 1:
                        raise ValueError('Unresolved fact ID reference.')
                    f = self.find(matches[0]['subject'], matches[0]['relation'])
                elif isinstance(ref, dict) and set(ref) == {'subject', 'relation'}:
                    f = self.find(ref['subject'], ref['relation'])
                else:
                    raise ValueError('Malformed fact reference.')
                return f.get('value', f.get('object')), f['value_type'], f.get('unit'), f
            raise ValueError('Unsupported operand reference.')
        return value, value_type, unit, None

    def evaluate(self, situation, p):
        if 'operator' not in p:
            return unknown('Legacy declaration needs explicit fact-based migration/configuration.')
        try:
            actual = self.find(p['subject'], p['relation'])
            value = p.get('expected_value', p.get('expected_object'))
            expected, expected_type, unit, expected_fact = self.operand(value, p['expected_value_type'], p.get('expected_unit'))
            if expected_type != p['expected_value_type']:
                raise ValueError('Referenced operand type differs from declaration.')
            a = actual.get('value', actual.get('object'))
            matched = compare(a, actual['value_type'], p['operator'], expected, expected_type,
                              actual.get('unit'), unit, p.get('reference_time'))
            evidence = actual['evidence'] + ('; ' + expected_fact['evidence'] if expected_fact else '')
            return {'evaluation': 'CONFIRMED' if matched else 'VIOLATED', 'evidence': evidence,
                    'reason': 'Typed operator ' + p['operator'] + (' satisfied.' if matched else ' violated.')}
        except (KeyError, TypeError, ValueError) as exc:
            return unknown(str(exc))

    def resolved_operands(self, p):
        try:
            a = self.find(p['subject'], p['relation'])
            b, t, u, f = self.operand(p.get('expected_value', p.get('expected_object')), p['expected_value_type'], p.get('expected_unit'))
            return {'actual': deepcopy(a), 'expected': {'value': b, 'value_type': t, 'unit': u,
                    'source_fact_id': f['fact_id'] if f else None}, 'operator': p['operator']}
        except (KeyError, TypeError, ValueError):
            return {'actual': None, 'expected': None, 'operator': p.get('operator')}
