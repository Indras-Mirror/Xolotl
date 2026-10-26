"""CPU guards for the byte budget, massive-expert protection and usage-weighted allocation."""
import unittest
from types import SimpleNamespace
import numpy as np
import lite_alloc as A


class Allocation(unittest.TestCase):
    def test_budget_and_protection(self):
        rng = np.random.default_rng(1017)
        c = np.cumsum(rng.uniform(.01, 1., (2, 384, 4)), axis=-1)
        c[..., 0] = 0
        protected = np.zeros((2, 384), bool)
        protected[:, 0] = True
        budget = 2 * A.BYTES['33'] + 766 * A.BYTES['22'] + 100 * (A.BYTES['33'] - A.BYTES['22'])
        k, size = A.solve(c, budget, protected)
        self.assertLessEqual(size, budget)
        self.assertTrue((k[protected] == 0).all())
        self.assertEqual(size, sum(A.BYTES[A.CLS[j]] for j in k.flat))

    def test_impossible_budget(self):
        with self.assertRaisesRegex(ValueError, 'feasible minimum'):
            A.solve(np.zeros((1, 384, 4)), 384 * A.BYTES['22'] - 1)
        with self.assertRaises(ValueError):
            A.solve(np.zeros((0, 384, 4)), 1)

    def test_massive_and_usage(self):
        ex = {str(i): dict(n=100, n_eval=20, y_energy=20., e33=.1, e23=.2, e32=.3, e22=.4) for i in range(384)}
        ex['0']['y_energy'] = 1e6
        ex['1']['n'] = 200
        ex['2']['n_eval'] = 0
        a = SimpleNamespace(cost='use', edge_weight=1., min_eval=16, massive=100.)
        _, c, unseen, protected = A.costs({0: ex}, a)
        self.assertTrue(protected[0, 0])
        self.assertTrue(unseen[0, 2])
        np.testing.assert_allclose(c[0, 1], 2 * c[0, 3])
        np.testing.assert_allclose(c[0, 2], c[0, 3])
        k, _ = A.solve(c, 384 * A.BYTES['33'], protected)
        self.assertTrue((k == 0).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
