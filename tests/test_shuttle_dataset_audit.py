"""Group-split constraints and deterministic frame-count balancing."""
import unittest

from src.shuttle.dataset_audit import select_split, validate_assignments


class SplitTests(unittest.TestCase):
    def test_known_optimum_and_determinism(self):
        counts = {'a': 70, 'b': 15, 'c': 15}
        first = select_split(counts, {'a'}, {'c'}, forced_val={'b'})
        self.assertEqual(first[0], {'a': 'train', 'b': 'val', 'c': 'test'})
        self.assertEqual(first[1], {'train': 70, 'val': 15, 'test': 15})
        self.assertEqual(first[2], 0)
        self.assertEqual(first, select_split(counts, {'a'}, {'c'}, forced_val={'b'}))

    def test_exposed_groups_cannot_be_test_candidates(self):
        with self.assertRaises(ValueError):
            select_split({'a': 70, 'b': 15, 'c': 15}, {'a'}, {'a', 'c'})
        with self.assertRaises(ValueError):
            select_split({'a': 70, 'b': 15, 'c': 15}, {'a'}, {'c'}, forced_val={'a'})

    def test_linked_sources_cannot_cross_splits(self):
        rows = [dict(scope='primary_48', source_group=g, leakage_block='linked',
                     v2_training_clip=0, test_eligibility='candidate_pending_source_verification')
                for g in ('a', 'b')]
        with self.assertRaises(ValueError):
            validate_assignments(rows, {'a': 'train', 'b': 'test'})
        validate_assignments(rows, {'a': 'val', 'b': 'val'})

    def test_trained_clip_must_stay_training(self):
        row = dict(scope='primary_48', source_group='a', leakage_block='a',
                   v2_training_clip=1, test_eligibility='historical_training')
        with self.assertRaises(ValueError):
            validate_assignments([row], {'a': 'val'})

    def test_development_clip_cannot_be_test(self):
        row = dict(scope='primary_48', source_group='a', leakage_block='a',
                   v2_training_clip=0, test_eligibility='historical_development')
        with self.assertRaises(ValueError):
            validate_assignments([row], {'a': 'test'})
