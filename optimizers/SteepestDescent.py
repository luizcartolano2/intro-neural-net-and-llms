import time
import numpy as np
from models.OptimizationResult import OptimizationResult
from optimizers.OptimizerBase import OptimizerBase


class SteepestDescent(OptimizerBase):
    def optimize(self, initial_values: np.ndarray) -> OptimizationResult:
        """
        Perform the steepest descent optimization using line search for step size.

        :param initial_values: Initial guess for the variables (NumPy array).
        :return: The optimized values of the variables.
        """
        x = np.array(initial_values, dtype=np.float64)
        start_time = time.time()

        for i in range(self.max_iters):
            self.log(f'Iteracao {i}')

            # Evaluate the function and the gradient at the current point
            func_value = self.function.evaluate_function_at(x)
            self.log(f'\tf(x) = {func_value}')
            grad = self.function.evaluate_gradient_at(x)
            self.log(f'\tgrad(x) = {grad}')

            # Save history for plotting
            self.history['values'].append(x.copy())
            self.history['func_values'].append(func_value)
            self.history['grad_magnitudes'].append(np.linalg.norm(grad))

            # Check for convergence
            if np.linalg.norm(grad) < self.tolerance:
                elapsed_time = time.time() - start_time
                self.log(f"Converged after {i + 1} iterations.")
                return OptimizationResult(x, i+1, elapsed_time)

            # --- Line search along the negative gradient direction ---
            alpha = 1.0
            c = 1e-4
            rho = 0.5
            while True:
                x_new = x - alpha * grad
                f_new = self.function.evaluate_function_at(x_new)
                if f_new <= func_value - c * alpha * np.linalg.norm(grad)**2:
                    break
                alpha *= rho
            self.log(f'\tLine search found alpha = {alpha}')

            # Update the variables
            self.log(f'\tx = x - alpha * grad = {x} - {alpha} * {grad} = {x_new}')
            x = x_new

            self.log("")

        elapsed_time = time.time() - start_time
        self.log(f"Reached maximum iterations ({self.max_iters}) without convergence.")
        return OptimizationResult(x, self.max_iters, elapsed_time)
