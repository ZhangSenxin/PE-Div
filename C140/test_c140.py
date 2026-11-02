"""Checks for threshold stopping rules, transitive families and logo units."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ESSI'))
from isc import right_isc
from analysis import hamming_matrix, components, family_data, select_examples
from figures import logo_matrix


class C140Tests(unittest.TestCase):
    def test_transitive_hamming_family(self):
        seqs = ['AAA', 'AAC', 'ACC', 'WWW']
        distances = hamming_matrix(seqs, 3)
        self.assertEqual(distances[0, 2], 2)
        self.assertEqual(components(distances), [[0, 1, 2], [3]])

    def test_asop_includes_overlap_and_ties_have_zero_direction(self):
        data = family_data({'exp1': {'AAA': 4, 'AAC': 3}, 'exp2': {'AAC': 5, 'ACC': 4}},
                           {'sequence_length': 3, 'minimum_plot_nodes': 1})
        family = data['families'][0]
        self.assertAlmostEqual(family['ASOP'], 1 / 3)
        self.assertEqual(family['signed_ASOP'], 0)
        self.assertEqual(family['exclusive_nodes'], 2)
        self.assertEqual(select_examples(data['families'], [0])[0]['cluster_id'], 'AA1_C0001')

    def test_isc_returns_preupdate_cutoff(self):
        values = np.r_[np.linspace(-1, 1, 100), 100]
        result, history = right_isc(values, .05, 1e-4, 100)
        self.assertTrue(result['converged'])
        self.assertAlmostEqual(result['k'], 1.959963984540054)
        self.assertEqual(result['threshold'], history[-1]['threshold'])
        self.assertEqual(int((values > result['threshold']).sum()), 1)

    def test_information_logo_limits(self):
        conserved = logo_matrix(['AAA', 'AAA'])
        uniform = logo_matrix([aa * 3 for aa in 'ACDEFGHIKLMNPQRSTVWY'])
        np.testing.assert_allclose(conserved['information_bits'], np.log2(20))
        np.testing.assert_allclose(uniform['information_bits'], 0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
