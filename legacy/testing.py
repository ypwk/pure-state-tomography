import unittest
from numpy import array, sqrt, linalg
from pure_state_tomography import tomography
import putils


class PreciseTesting(unittest.TestCase):
    def setUp(self):
        self.talg = tomography()

    def run_precise_test(self, vector):
        res = self.talg.pure_state_tomography(
            input_state=vector,
            n_qubits=putils.fast_log2(len(vector)),
            precise=True,
            simulator=True,
            verbose=False,
        )
        self.assertAlmostEqual(0, linalg.norm(vector - res), places=9)

    def test_vector1(self):
        state = array([1 / 2, 1 / sqrt(2), 1 / sqrt(6), 1 / sqrt(12)])
        self.run_precise_test(state)

    def test_vector2(self):
        state = array([1 / 2, -1 / sqrt(2), 1 / sqrt(6), 1 / sqrt(12)])
        self.run_precise_test(state)

    def test_vector3(self):
        state = array([1 / 2, 0, -2 / sqrt(6), 1 / sqrt(12)])
        self.run_precise_test(state)

    def test_vector4(self):
        state = array([1 / 2, 0, 0, -3 / sqrt(12)])
        self.run_precise_test(state)


class ShotsTesting(unittest.TestCase):
    def setUp(self):
        self.talg = tomography()

    def run_shots_test(self, vector):
        res = self.talg.pure_state_tomography(
            input_state=vector,
            n_qubits=putils.fast_log2(len(vector)),
            precise=False,
            simulator=True,
            verbose=False,
            n_shots=putils.fast_pow(2, 10)
        )
        self.assertLessEqual(linalg.norm(vector - res), 0.20)

    def test_vector1(self):
        state = array([1 / 2, 1 / sqrt(2), 1 / sqrt(6), 1 / sqrt(12)])
        self.run_shots_test(state)

    def test_vector2(self):
        state = array([1 / 2, -1 / sqrt(2), 1 / sqrt(6), 1 / sqrt(12)])
        self.run_shots_test(state)

    def test_vector3(self):
        state = array([1 / 2, 0, -2 / sqrt(6), 1 / sqrt(12)])
        self.run_shots_test(state)

    def test_vector4(self):
        state = array([1 / 2, 0, 0, -3 / sqrt(12)])
        self.run_shots_test(state)


if __name__ == "__main__":
    unittest.main()
