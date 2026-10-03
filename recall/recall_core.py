"""Small, deterministic outcome-backed memory. No model weights are changed."""
from copy import deepcopy
import math
import re
import unicodedata
from applicability import LocalEvaluator, check_applicability, validate_preconditions
from fact_extraction import FactExtractor
from fact_applicability import FactEvaluator, bindings, render_action, validate_action

TRANSFER_THRESHOLD = 0.50
SUCCESS_SCORE_THRESHOLD = 0.50


def normalize(value):
    return unicodedata.normalize('NFC', value.strip())


def features_for(text):
    text = unicodedata.normalize('NFC', normalize(text).casefold())
    return sorted(set(re.findall(r'[^\W_]+', text, flags=re.UNICODE)))


def canonical_features(value):
    """Explicit caller declarations; no inference from facts or expectations."""
    if (not isinstance(value, list) or not 1 <= len(value) <= 200 or
            any(not isinstance(f, str) or not f.strip() or len(f) > 2000 for f in value)):
        raise ValueError('Features must be a list of 1 to 200 nonempty strings, each at most 2,000 characters.')
    return sorted(set(normalize(f).casefold() for f in value))


def similarity(left, right):
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a | b else 0.0


def critical_mismatch(action, original, current):
    """Detect explicit action prohibitions/unavailability, not general semantics.

    Supported local clauses: 'откат [версии] недоступен',
    'недоступен откат [версии]', 'нельзя откатить ...'.
    The vocabulary describes constraints, never particular services or objects.
    """
    negative = r'(?:не\s*доступ(?:ен|на|но|ны)|не\s*возмож(?:ен|на|но|ны)|запрещ[её]н(?:а|о|ы)?)'

    def stem(word):
        return re.sub(r'(?:ить|ать|ять|ы|а|у|ом|е)$', '', word)

    operators = {'выполнить', 'сделать', 'провести', 'осуществить'}

    def english_terms(text):
        return re.sub(r'\broll\s+back\b', 'rollback', text.casefold())

    def restrictions(text):
        result = {}
        for clause in re.split(r'[.!?;\n,]+', normalize(text).casefold()):
            clause = clause.strip()
            if re.search(r'\bне\s+(?:запрещ[её]н\w*|невозмож\w*|недоступ\w*)\b', clause):
                continue
            # Anchoring avoids turning 'не запрещён' into a prohibition.
            match = re.fullmatch(r'(?:нельзя|невозможно|запрещено)\s+'
                                 r'(?:(?:выполнить|сделать|провести|осуществить)\s+)?'
                                 r'(\w+)(?:\s+.+)?', clause)
            if not match:
                match = re.fullmatch(rf'{negative}\s+(\w+)(?:\s+[^.!?;]+)?', clause)
            if not match:
                match = re.fullmatch(rf'(\w+)(?:\s+\w+){{0,2}}\s+{negative}', clause)
            if match and match[1] not in {'не', 'нет', 'если'}:
                result[stem(match[1])] = clause
            english = english_terms(clause)
            constraint = r'(?:unavailable|not available|impossible|not possible|prohibited|forbidden|disabled|not permitted|not allowed)'
            match = re.fullmatch(rf'(?:the\s+)?(\w+)(?:\s+(?:is|are))?\s+{constraint}', english)
            if not match:
                match = re.fullmatch(r'(?:do not|cannot|can not|must not)\s+'
                                     r'(?:(?:perform|execute)\s+)?(\w+)(?:\s+.+)?', english)
            if match:
                result[stem(match[1])] = clause
        return result

    previous = restrictions(original)
    action_words = {stem(word) for word in features_for(english_terms(action)) if word not in operators}
    for subject, clause in restrictions(current).items():
        if subject in action_words and subject not in previous:
            return f'New constraint on the action: “{clause}”.'
    return None


def adapt_transferred_action(action, original, current):
    """Bind an explicit service name only when the surrounding situation agrees.

    Free-form or ambiguous object descriptions keep the confirmed action intact.
    Only service-qualified names in the action are changed, never bare mentions.
    """
    service = re.compile(
        r'(?<!\w)(?i:сервис(?:а|у|ом|е)?|services?)\s+'
        r'(?P<name>«[^«»\n]+»|"[^"\n]+"|[A-ZА-ЯЁ][\w-]*)(?![\w-])')
    before, after = list(service.finditer(original)), list(service.finditer(current))
    if len(before) != 1 or len(after) != 1:
        return action
    old, new = before[0], after[0]
    # Do not interpret a list or an unquoted multiword name as one object.
    continuation = re.compile(r'^\s*(?:[,/&+]|(?:и|или|and|or)\b|[A-ZА-ЯЁ])')
    if any(continuation.match(text[match.end():])
           for text, match in ((original, old), (current, new))):
        return action

    def context(text, match):
        return ' '.join((text[:match.start('name')] + '\0' +
                         text[match.end('name'):]).casefold().split())

    if context(original, old) != context(current, new):
        return action
    matches = list(service.finditer(action))
    if len(matches) != 1 or matches[0]['name'] != old['name']:
        return action
    target = matches[0]
    if continuation.match(action[target.end():]):
        return action
    return action[:target.start('name')] + new['name'] + action[target.end('name'):]


class Recall:
    def __init__(self, model, evaluator=None, fact_extractor=None, action_renderer=None, semantic_registry=None):
        self.model = model
        self.evaluator = evaluator
        self.fact_extractor = fact_extractor if fact_extractor is not None else FactExtractor()
        if semantic_registry is not None and fact_extractor is None:
            from pers_a_semantics import PersASemanticExtractor as CanonicalExtractor
            self.fact_extractor = CanonicalExtractor(semantic_registry)
        self.action_renderer = action_renderer if action_renderer is not None else render_action
        self._state = dict(situations=[], decisions=[], outcomes=[], experiences=[])

    def snapshot(self):
        return deepcopy(self._state)

    def decide(self, text, *, facts=None, features=None):
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2000:
            raise ValueError('Enter a situation between 1 and 2,000 characters long.')
        text = text.strip()
        features = features_for(text) if features is None else canonical_features(features)
        if not features or len(features) > 200:
            raise ValueError('The situation must contain between 1 and 200 distinct words or numbers.')
        if len(self._state['decisions']) >= 100:
            raise ValueError('This session has reached the limit of 100 decisions.')
        if facts is not None and not isinstance(facts, list):
            raise ValueError('Structured facts must be a list.')
        evaluator_situation = text if facts is None else {'text': text, 'facts': deepcopy(facts)}
        candidates = []
        for exp in self._state['experiences']:
            score = similarity(features, exp['situation_features'])
            if exp['success'] and exp['score'] >= SUCCESS_SCORE_THRESHOLD and score >= TRANSFER_THRESHOLD:
                candidates.append((score, exp))
        evidence = None
        transferred = False
        if candidates:
            score, exp = min(candidates, key=lambda item: (
                -item[0], -item[1]['score'], normalize(item[1]['action']), item[1]['id']))
            original = next((s['text'] for s in self._state['situations']
                             if s['id'] == exp['situation_id']), '')
            runtime_exp = (self.fact_extractor.prepare_experience(exp)
                           if hasattr(self.fact_extractor, 'prepare_experience') else exp)
            extraction = self.fact_extractor.extract(text, facts)
            if getattr(self.fact_extractor, 'mode', None) == 'canonical':
                self.fact_extractor.last_trace = extraction
                extraction['call_counters']['fallback_attempts'] = 0
                extraction['memory_before'] = {k: deepcopy(self._state[k]) for k in ['experiences', 'outcomes']}
            structured_action = validate_action(runtime_exp.get('structured_action'))
            bound, binding_checks = bindings(structured_action, extraction)
            if getattr(self.fact_extractor, 'mode', None) == 'canonical':
                for binding, check in zip(bound, binding_checks):
                    if any(r['fact'].get('relation') == 'role' and
                           r['fact'].get('value', r['fact'].get('object')) == binding['role']
                           for r in extraction['rejected_facts']):
                        check.update(evaluation='UNKNOWN', reason='Relevant role assertion failed evidence verification.')
            evaluator = self.evaluator if self.evaluator is not None else FactEvaluator(extraction, bound)
            applicability = check_applicability(evaluator_situation, runtime_exp.get('preconditions'), evaluator, binding_checks)
            if isinstance(evaluator, FactEvaluator):
                for row in applicability['evaluated_preconditions']:
                    if row['precondition'].get('operator'):
                        row['resolved_operands'] = evaluator.resolved_operands(row['precondition'])
            applicability.update(extraction=extraction, entity_bindings=bound,
                                 resolved_parameterized_action=None)
            if applicability['decision'] == 'REJECT':
                evidence = dict(experience=deepcopy(exp), similarity=score,
                                score=exp['score'], status=('TRANSFER BLOCKED: critical mismatch'
                                if applicability['applicability'] == 'VIOLATED' else
                                'TRANSFER BLOCKED: applicability unknown'),
                                reason=applicability['reason'], applicability=applicability)
            else:
                resolved, action = self.action_renderer(structured_action, bound)
                applicability['resolved_parameterized_action'] = resolved
                reason = f"Reused confirmed experience #{exp['id']}. Similarity: {score:.0%}."
                source = 'EXPERIENCE'
                evidence = dict(experience=deepcopy(exp), similarity=score,
                                score=exp['score'], action=action, applicability=applicability)
                transferred = True
        if not transferred:
            # Validate even injected model adapters, before any state mutation.
            from model_client import ModelError, validate_output
            if evidence and getattr(self.fact_extractor, 'mode', None) == 'canonical':
                evidence['applicability']['extraction']['call_counters']['fallback_attempts'] += 1
            try:
                answer = validate_output(self.model.decide(text, features), features)
            except ModelError as exc:
                if evidence and evidence.get('status'):
                    raise ModelError(f"{evidence['status']}. {evidence['reason']} {exc}") from exc
                raise
            features, action, reason = answer['features'], answer['action'], answer['reason']
            source = 'MODEL'
        identifier = len(self._state['decisions']) + 1
        situation = dict(id=identifier, text=text, features=features)
        if facts is not None:
            situation['facts'] = deepcopy(facts)
        decision = dict(id=identifier, situation_id=identifier, action=action,
                        source=source, reason=reason, evidence=evidence)
        if evidence and getattr(self.fact_extractor, 'mode', None) == 'canonical':
            extraction['final_decision'] = 'TRANSFER' if transferred else 'REJECT'
            extraction['memory_after'] = {k: deepcopy(self._state[k]) for k in ['experiences', 'outcomes']}
        self._state['situations'].append(situation)
        self._state['decisions'].append(decision)
        return deepcopy(decision)

    def save_outcome(self, decision_id, success, score, feedback, preconditions=None, structured_action=None):
        if type(decision_id) is not int or not 1 <= decision_id <= len(self._state['decisions']):
            raise ValueError('Select an existing decision.')
        if type(success) is not bool:
            raise ValueError('Success must be either checked or unchecked.')
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Enter a numeric score between 0 and 1.')
        if not isinstance(feedback, str) or len(feedback) > 1000:
            raise ValueError('Feedback must be no longer than 1,000 characters.')
        if any(o['decision_id'] == decision_id for o in self._state['outcomes']):
            raise ValueError('An outcome has already been saved for this decision.')
        declarations = validate_preconditions(preconditions)
        action_declaration = validate_action(structured_action)
        decision = self._state['decisions'][decision_id - 1]
        situation = self._state['situations'][decision['situation_id'] - 1]
        outcome = dict(decision_id=decision_id, success=success, score=float(score), feedback=normalize(feedback))
        experience = dict(id=len(self._state['experiences']) + 1,
                          situation_id=situation['id'], decision_id=decision_id,
                          situation_features=deepcopy(situation['features']),
                          action=decision['action'], success=success, score=float(score), preconditions=declarations,
                          structured_action=action_declaration)
        self._state['outcomes'].append(outcome)
        self._state['experiences'].append(experience)
        return deepcopy(experience)
