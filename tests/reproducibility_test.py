import random
import unittest

import numpy as np
import torch

from text2fx.__main__ import seed_runtime


class SeedRuntimeTests(unittest.TestCase):
    def test_seed_controls_every_rng_used_by_optimization(self):
        seed_runtime(1729)
        first = (random.random(), np.random.random(), torch.rand(3))

        seed_runtime(1729)
        second = (random.random(), np.random.random(), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))


if __name__ == "__main__":
    unittest.main()
