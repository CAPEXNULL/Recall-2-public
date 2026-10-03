"""Typed comparisons; labels/categories/action types never select decision rules."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

CONFIG = json.loads((Path(__file__).with_name('semantic_config.json')).read_text(encoding='utf-8'))
TYPES = {'boolean', 'number', 'string/entity', 'datetime', 'set/list'}
OPERATORS = {'EQ', 'NE', 'LT', 'LTE', 'GT', 'GTE', 'IN', 'NOT_IN', 'CONTAINS', 'CONTAINS_ALL',
             'BEFORE', 'AFTER', 'AT_OR_BEFORE', 'AT_OR_AFTER', 'MAX_AGE'}


def canonical(value, value_type, unit=None):
    if value_type == 'boolean':
        if type(value) is not bool:
            raise ValueError('Boolean type required.')
        return value
    if value_type == 'number':
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Finite number required, not boolean.')
        if unit is None:
            return float(value)
        if unit not in CONFIG['units']:
            raise ValueError('Unsupported unit.')
        _, scale, offset = CONFIG['units'][unit]
        return value * scale + offset
    if value_type == 'datetime':
        if not isinstance(value, str):
            raise ValueError('ISO datetime required.')
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError('Ambiguous timezone.')
        return parsed.astimezone(timezone.utc).isoformat()
    if value_type == 'string/entity':
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Nonempty string/entity required.')
        return value.strip().casefold()
    if value_type == 'set/list':
        if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
            raise ValueError('A list of nonempty strings required.')
        return sorted({v.strip().casefold() for v in value})
    raise ValueError('Unsupported value type.')


def dimension(unit):
    if unit is None:
        return None
    if unit not in CONFIG['units']:
        raise ValueError('Unsupported unit.')
    return CONFIG['units'][unit][0]


def compare(actual, actual_type, operator, expected, expected_type, actual_unit=None,
            expected_unit=None, reference_time=None):
    if operator not in OPERATORS:
        raise ValueError('Unsupported operator.')
    if operator == 'CONTAINS_ALL':
        if actual_type != 'set/list' or expected_type != 'set/list' or actual_unit is not None or expected_unit is not None:
            raise ValueError('CONTAINS_ALL requires two unitless sets.')
        return set(canonical(expected, expected_type)) <= set(canonical(actual, actual_type))
    if operator == 'MAX_AGE':
        if actual_type != 'datetime' or expected_type != 'number' or reference_time is None:
            raise ValueError('MAX_AGE needs datetime, duration and fixed reference time.')
        a = datetime.fromisoformat(canonical(actual, actual_type))
        ref = datetime.fromisoformat(canonical(reference_time, 'datetime'))
        if dimension(expected_unit) != 'duration':
            raise ValueError('Duration unit required.')
        limit = canonical(expected, 'number', expected_unit)
        age = (ref - a).total_seconds()
        if limit < 0 or age < 0:
            raise ValueError('Negative duration or future observation.')
        return age <= limit
    if operator in {'IN', 'NOT_IN', 'CONTAINS'}:
        left_type, right_type = ('set/list', 'string/entity') if operator == 'CONTAINS' else ('string/entity', 'set/list')
        if actual_type != left_type or expected_type != right_type:
            raise ValueError('Membership types incompatible.')
        a, b = canonical(actual, actual_type), canonical(expected, expected_type)
        found = b in a if operator == 'CONTAINS' else a in b
        return not found if operator == 'NOT_IN' else found
    if actual_type != expected_type:
        raise ValueError('Operand types differ.')
    if actual_type == 'number' and dimension(actual_unit) != dimension(expected_unit):
        raise ValueError('Incompatible units.')
    a, b = canonical(actual, actual_type, actual_unit), canonical(expected, expected_type, expected_unit)
    if operator in {'BEFORE', 'AFTER', 'AT_OR_BEFORE', 'AT_OR_AFTER'}:
        if actual_type != 'datetime':
            raise ValueError('Temporal operator requires datetime.')
        a, b = datetime.fromisoformat(a), datetime.fromisoformat(b)
        operator = {'BEFORE': 'LT', 'AFTER': 'GT', 'AT_OR_BEFORE': 'LTE', 'AT_OR_AFTER': 'GTE'}[operator]
    elif operator in {'LT', 'LTE', 'GT', 'GTE'} and actual_type != 'number':
        raise ValueError('Ordered comparison requires number or temporal operator.')
    if operator in {'EQ', 'NE'}:
        eq = math.isclose(a, b, rel_tol=0, abs_tol=CONFIG['numeric_tolerance']) if actual_type == 'number' else a == b
        return eq if operator == 'EQ' else not eq
    return {'LT': lambda: a < b, 'LTE': lambda: a <= b, 'GT': lambda: a > b, 'GTE': lambda: a >= b}[operator]()
