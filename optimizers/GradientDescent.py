import time
import numpy as np

from models.OptimizationResult import OptimizationResult
from optimizers.OptimizerBase import OptimizerBase


class GradientDescent(OptimizerBase):
    def optimize(self, initial_values: np.ndarray) -> OptimizationResult:
        """
        Perform gradient descent optimization.

        :param initial_values: Initial guess for the variables (NumPy array).
        :return: The optimized values of the variables and the function value at the optimized point.
        """
        x = np.array(initial_values, dtype=np.float64)
        start_time = time.time()

        for i in range(self.max_iters):
            self.log(f'Iteration {i}')
            # Evaluate the function and the gradient at the current point
            func_value = self.function.evaluate_function_at(x)
            self.log(f'\tf(x) = {func_value}')
            grad = self.function.evaluate_gradient_at(x)
            self.log(f'\tgrad(x) = {grad}')

            # Save history for plotting
            self.history['values'].append(x.copy())
            self.history['func_values'].append(func_value)
            self.history['grad_magnitudes'].append(np.linalg.norm(grad))

            # Update the variables using gradient descent rule
            self.log(f'\tx = x - lr * grad = {x} - {self.learning_rate} * {grad} = {x - self.learning_rate * grad}')
            x = x - self.learning_rate * grad

            # Stop if the gradient is small enough (converged)
            if np.linalg.norm(grad) < self.tolerance:
                elapsed_time = time.time() - start_time
                self.log(f"Converged after {i + 1} iterations.")
                return OptimizationResult(x, i+1, elapsed_time)

            self.log("")

        elapsed_time = time.time() - start_time
        self.log(f"Reached maximum iterations ({self.max_iters}) without convergence.")

        return OptimizationResult(x, self.max_iters, elapsed_time)
