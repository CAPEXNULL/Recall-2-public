"""Offline API demo: scripted candidates, real deterministic production validation."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'recall'))
from pers_a_semantics import PersASemanticExtractor
from recall_core import Recall

REGISTRY = dict(roles=[], entities=[dict(id='node', aliases=[])], relations=[
    dict(id='ready', meaning='explicit current ready state', value_type='boolean', semantic_class='PROPERTY')])

class OfflineRecommendation:
    def decide(self, text, features):
        return dict(features=features, action='Inspect node.', reason='Offline recommendation fixture; no NVIDIA call.')

def candidates(purpose, request):
    assert purpose == 'extraction'
    # Deliberately propose false for absent data: production code must reject it.
    value = 'equal to true' in current_text
    hypothesis = dict(subject_text='node', relation_id='ready', value=value,
                      unit=None, span_id='s1', polarity=True, assertion_mode='ACTUAL')
    return dict(model=request['model'], choices=[dict(finish_reason='stop', message=dict(
        content=json.dumps(dict(hypotheses=[hypothesis]))))])

extractor = PersASemanticExtractor(REGISTRY, candidates)
core = Recall(OfflineRecommendation(), fact_extractor=extractor)
seed = core.decide('Seed observation.', features=['inspection'])
core.save_outcome(seed['id'], True, 1.0, 'Reported successful inspection.', preconditions=[
    dict(id='ready-required', category='Availability', subject='node', relation='ready',
         operator='EQ', mandatory=True, expected_value=True, expected_value_type='boolean')],
    structured_action=dict(action_type='inspect', parameters=[], display_template='Inspect node.'))

for current_text, expected in [
    ('node has ready equal to true.', ('CONFIRMED', 'TRANSFER')),
    ('node has ready equal to false.', ('VIOLATED', 'REJECT')),
    ('node current state is absent.', ('UNKNOWN', 'REJECT'))]:
    decision = core.decide(current_text, features=['inspection'])
    result = decision['evidence']['applicability']
    actual = result['applicability'], result['decision']
    assert actual == expected, (actual, expected)
    print(current_text, '->', *actual)
print('PASS: production API, offline fixtures, zero network calls.')
