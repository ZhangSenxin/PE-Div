"""Small numerical checks; no project reads or large result files required."""
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from multi_round_pipeline import single_round, combine, translate_dna, read_counts


class ESSITests(unittest.TestCase):
    def test_single_comparison_formula(self):
        rows, info = single_round(Counter(A=3, B=1), Counter(A=1, B=5))
        expected_s = math.log2((2 / 8) / (4 / 6))
        expected_v = (1 / 2 + 1 / 4) / math.log(2)**2
        self.assertEqual(info['K'], 2)
        self.assertAlmostEqual(rows['A']['S_i'], expected_s)
        self.assertAlmostEqual(rows['A']['sigma_i']**2, expected_v)
        self.assertAlmostEqual(rows['A']['Z_i'], expected_s / math.sqrt(expected_v))

    def test_dual_absent_keeps_joint_weight_and_adds_variance(self):
        r1, i1 = single_round(Counter(A=100, B=1), Counter(A=1, B=100))
        r2, i2 = single_round(Counter(C=2, D=1), Counter(C=1, D=2))
        plain, plain_info = combine([r1, r2], [i1, i2], 'none', 2)
        adjusted, adjusted_info = combine([r1, r2], [i1, i2], 'legacy', 2)
        plain = {r['label']: r for r in plain}
        adjusted = {r['label']: r for r in adjusted}
        self.assertEqual(plain_info['tau2_total'], adjusted_info['tau2_total'])
        self.assertEqual(plain['B']['shrinkage_weight'], adjusted['B']['shrinkage_weight'])
        expected_delta_v = (1 + 1 / 101) / math.log(2)**2
        self.assertAlmostEqual(adjusted['B']['sigma_total']**2 - plain['B']['sigma_total']**2, expected_delta_v)
        self.assertAlmostEqual(adjusted['B']['theta_total'], adjusted['B']['shrinkage_weight'] * adjusted['B']['S_total'])

    def test_translation_and_synonymous_pooling(self):
        self.assertEqual(translate_dna('atggct'), ('MA', None))
        self.assertEqual(translate_dna('ATGTAA')[1], 'stop_codon')
        self.assertEqual(translate_dna('ATGA')[1], 'invalid_length')
        self.assertEqual(translate_dna('ATGNNN')[1], 'invalid_bases')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reads.txt'
            path.write_text('ID\tSeq\n0\tGCT\n0\tGCC\n1\tGCT\n2\tTAA\n', encoding='utf-8')
            counts, skipped = read_counts(path, set(), False, 'dna')
            self.assertEqual(counts, Counter(A=3))
            self.assertEqual(skipped, 1)


if __name__ == '__main__':
    unittest.main()
