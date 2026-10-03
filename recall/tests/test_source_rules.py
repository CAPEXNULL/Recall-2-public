from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import unittest
from pers_a_semantics import PersASemanticExtractor
from source_assertions import numeric_identity

REG=dict(roles=[],relations=[dict(id='ready',meaning='explicit current ready state',value_type='boolean',semantic_class='PROPERTY'),
    dict(id='amount',meaning='explicit measured duration',value_type='number',semantic_class='PROPERTY',unit='s',dimension='duration'),
    dict(id='members',meaning='complete reported member set',value_type='set/list',semantic_class='PROPERTY')],entities=[dict(id='node',aliases=[])])

def extract(text,values,relation='ready',context=None,mode='ACTUAL'):
    calls=[]
    def provider(purpose,req):
        calls.append(purpose)
        assert purpose=='extraction','Models must not verify states'
        rows=[dict(subject_text='node',relation_id=relation,value=v,unit=u,span_id='d0' if context else 'range' if '. ' in text else 's1',polarity=True,assertion_mode=mode) for v,u in values]
        return dict(model=req['model'],choices=[dict(finish_reason='stop',message=dict(content=json.dumps(dict(hypotheses=rows))))])
    t=PersASemanticExtractor(REG,provider).extract(text,context)
    assert calls==['extraction']
    return t

class SourceRules(unittest.TestCase):
    def test_missing_is_not_false(self):
        self.assertEqual(extract('node current state is absent.',[(False,None)])['facts'],[])
    def test_modal_states_are_unknown(self):
        for text in ['node will be ready.','An operator predicts node is ready.','node might be ready.','node was ready.']:
            self.assertEqual(extract(text,[(True,None)])['facts'],[])
    def test_negating_missing_data_does_not_supply_value(self):
        self.assertEqual(extract('node data is not absent.',[(False,None)])['facts'],[])
    def test_explicit_false(self):
        self.assertIs(extract('node has ready equal to false.',[(False,None)])['facts'][0]['value'],False)
    def test_explicit_negative_current_state(self):
        self.assertIs(extract('node is not ready.',[(False,None)])['facts'][0]['value'],False)
    def test_model_modality_cannot_override_explicit_source(self):
        self.assertIs(extract('node is ready.',[(True,None)],mode='FORECAST')['facts'][0]['value'],True)
    def test_claim_disagreement_is_unknown(self):
        self.assertEqual(extract('node is ready.',[(False,None)])['facts'],[])
    def test_numeric_identity_is_exact(self):
        self.assertEqual(len({numeric_identity(v) for v in [90,90.0,90.00]}),1)
        self.assertNotEqual(numeric_identity(90),numeric_identity(90.0000000001))
        self.assertNotEqual(numeric_identity(10**35),numeric_identity(10**35+1))
    def test_canonical_units(self):
        self.assertEqual(extract('The measured amount of node is 1.5 min.',[(90,'s')],'amount')['facts'][0]['value'],90)
    def test_omitted_collection_is_not_empty(self):
        self.assertEqual(extract('The members field of node was omitted.',[([],None)],'members')['facts'],[])
    def test_explicit_empty_collection(self):
        self.assertEqual(extract('The complete members set for node is [].',[([],None)],'members')['facts'][0]['value'],[])
    def test_explicit_zero(self):
        self.assertEqual(extract('The measured amount of node is 0 s.',[(0,'s')],'amount')['facts'][0]['value'],0)
    def test_conflict_remains_unknown(self):
        t=extract('node is ready. node is not ready.',[(True,None),(False,None)])
        self.assertEqual(t['semantic_groups'][0]['state'],'CONFLICT');self.assertEqual(len(t['facts']),2)
    def test_structured_source_types_and_modality(self):
        for value,mode in [(None,'ACTUAL'),(False,'FORECAST')]:
            datum=dict(subject='node',relation='ready',value=value,assertion_mode=mode)
            self.assertEqual(extract('node',[(False,None)],context=[datum])['facts'],[])
        datum=dict(subject='node',relation='ready',value=False)
        self.assertIs(extract('node',[(False,None)],context=[datum])['facts'][0]['value'],False)
    def test_unrecognised_language_fails_closed(self):
        self.assertEqual(extract('node: состояние не подтверждено.',[(False,None)])['facts'],[])
    def test_candidate_cannot_cross_subjects(self):
        self.assertEqual(extract('The ready value for "node spare" is false.',[(False,None)])['facts'],[])

if __name__=='__main__':unittest.main()
