"""Production interface and set-contract tests on independent local fixtures."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recall_core import Recall, features_for
from fact_model import compare


class Model:
    def __init__(self):
        self.inputs = []

    def decide(self, text, features):
        self.inputs.append((text, deepcopy(features)))
        return dict(features=features, action='Inspect the apparatus.', reason='Local fixture.')


class Extractor:
    def __init__(self):
        self.inputs = []

    def extract(self, text, facts):
        self.inputs.append((text, facts))
        return {'facts': [], 'rejected_facts': []}


class ExplicitFeatures(unittest.TestCase):
    def test_default_and_canonical_explicit_input(self):
        model = Model()
        core = Recall(model)
        core.decide('Ordinary observation.')
        self.assertEqual(model.inputs[-1][1], features_for('Ordinary observation.'))
        declared = [' CAFe\u0301 ', 'café', 'WARM']
        original = deepcopy(declared)
        core.decide('A wholly separate source.', features=declared)
        self.assertEqual(model.inputs[-1][1], ['café', 'warm'])
        self.assertEqual(core.snapshot()['situations'][-1]['features'], ['café', 'warm'])
        self.assertEqual(declared, original)

    def test_invalid_input_is_atomic(self):
        for declared in ([], 'warm', [False], [None], [' '], ['x'] * 201, ['x' * 2001]):
            with self.subTest(declared=repr(declared)[:60]):
                model = Model()
                core = Recall(model)
                before = core.snapshot()
                with self.assertRaises(ValueError):
                    core.decide('Source.', features=declared)
                self.assertEqual(core.snapshot(), before)
                self.assertEqual(model.inputs, [])

    def seeded(self):
        model, extractor = Model(), Extractor()
        core = Recall(model, fact_extractor=extractor)
        seed = core.decide('Seed observation.', features=['alpha', 'beta'])
        core.save_outcome(seed['id'], True, 1, 'Confirmed', preconditions=[
            {'id': 'p', 'category': 'Availability', 'requirement': 'local', 'mandatory': True, 'expected': True}])
        return core, model, extractor

    def test_admission_uses_features_but_extraction_keeps_source(self):
        core, model, extractor = self.seeded()
        core.decide('Entirely different literal source.', features=['alpha', 'beta'])
        self.assertEqual(extractor.inputs, [('Entirely different literal source.', None)])
        self.assertEqual(len(model.inputs), 2)  # Unknown binding still requires fallback.

    def test_features_do_not_bypass_threshold(self):
        core, model, extractor = self.seeded()
        core.decide('Different source.', features=['unrelated'])
        self.assertEqual(extractor.inputs, [])
        self.assertEqual(len(model.inputs), 2)

    def test_contains_all(self):
        self.assertTrue(compare(['BLUE', 'green', 'blue'], 'set/list', 'CONTAINS_ALL', ['blue'], 'set/list'))
        self.assertFalse(compare(['blue'], 'set/list', 'CONTAINS_ALL', ['blue', 'green'], 'set/list'))
        self.assertTrue(compare([], 'set/list', 'CONTAINS_ALL', [], 'set/list'))
        self.assertTrue(compare(['a', 'b'], 'set/list', 'CONTAINS_ALL', ['b', 'a'], 'set/list'))
        with self.assertRaises(ValueError):
            compare(['blue'], 'set/list', 'CONTAINS_ALL', 'blue', 'string/entity')
        with self.assertRaises(ValueError):
            compare(['blue'], 'set/list', 'CONTAINS_ALL', ['blue'], 'set/list', actual_unit='s')


if __name__ == '__main__':
    unittest.main()
