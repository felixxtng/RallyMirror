"""Check pooled detection metrics use counts rather than macro percentages."""
import unittest
from src.shuttle.batch_evaluate import pooled_counts


class PoolingTests(unittest.TestCase):
    def test_micro_counts_and_ratios(self):
        result = pooled_counts([dict(tp=1, fp=99, fn=0, tn=2),
                                dict(tp=9, fp=1, fn=10, tn=3)])
        self.assertEqual([result[k] for k in ('tp', 'fp', 'fn', 'tn')], [10, 100, 10, 5])
        self.assertAlmostEqual(result['precision'], 10/110)
        self.assertAlmostEqual(result['recall'], 10/20)
        self.assertAlmostEqual(result['f1'], 20/130)
        self.assertNotAlmostEqual(result['precision'], (.01+.9)/2)

    def test_undefined_ratios(self):
        self.assertEqual(pooled_counts([]), dict(tp=0, fp=0, fn=0, tn=0,
                                               precision=None, recall=None, f1=None))
