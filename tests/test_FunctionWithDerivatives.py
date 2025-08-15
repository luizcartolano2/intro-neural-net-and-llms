import unittest
import sympy as sp
import numpy as np
from models.FunctionWithDerivatives import FunctionWithDerivatives


class TestFunctionWithDerivatives(unittest.TestCase):
    """
    Unit tests for the FunctionWithDerivatives class.
    Tests function evaluation, gradient, and Hessian computations.

    Methods:
        setUp(): Sets up the test case with a sample function.
        test_evaluate_function_at(): Tests function evaluation at a point.
        test_evaluate_gradient_at(): Tests gradient evaluation at a point.
        test_evaluate_hessian_at(): Tests Hessian evaluation at a point.
    """
    def setUp(self):
        """
        Set up a sample function for testing: f(x1, x2) = x1^2 + x2^2
        :return: None
        """
        x1, x2 = sp.symbols('x1 x2')
        self.symbols = (x1, x2)
        self.func_expr = x1**2 + x2**2
        self.function = FunctionWithDerivatives(self.symbols, self.func_expr)

    def test_evaluate_function_at(self):
        """
        Test function evaluation at the point (1, 2).
        :return: None
        """
        result = self.function.evaluate_function_at([1, 2])
        expected = 1**2 + 2**2  # 5
        self.assertAlmostEqual(result, expected, places=5)

    def test_evaluate_gradient_at(self):
        """
        Test gradient evaluation at the point (1, 2).
        :return: None
        """
        result = self.function.evaluate_gradient_at([1, 2])
        expected = np.array([2 * 1, 2 * 2])  # [2, 4]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_evaluate_hessian_at(self):
        """
        Test Hessian evaluation at the point (1, 2).
        :return: None
        """
        result = self.function.evaluate_hessian_at([1, 2])
        expected = np.array([[2, 0], [0, 2]])  # [[2, 0], [0, 2]]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)


if __name__ == '__main__':
    unittest.main()
