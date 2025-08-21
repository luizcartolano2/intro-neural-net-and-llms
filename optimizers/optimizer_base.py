from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

from models.function_with_derivatives import FunctionWithDerivatives
from models.optimization_result import OptimizationResult


class OptimizerBase(ABC):
    """
    Abstract base class for optimization algorithms.
    This class defines the interface for optimization algorithms,
    including methods for optimization, plotting convergence trajectories,
    and plotting convergence metrics.

    Attributes:
        function: An instance of FunctionWithDerivatives that provides the function to optimize.
        learning_rate: The step size for each update in the optimization process.
        max_iters: Maximum number of iterations for the optimization.
        tolerance: The stopping criterion based on the magnitude of the gradient.
        history: A dictionary to store the history of values, function values, and gradient magnitudes during optimization.
        verbose: If True, prints detailed information during optimization.

    Methods:
        optimize(initial_values): Abstract method to perform optimization, must be implemented by subclasses.
        plot_convergence_trajectory(x_range, y_range, resolution, levels): Plots the trajectory of the optimization on the contour plot of the function.
        plot_convergence_metrics(): Plots the function values and gradient magnitudes over iterations.
    """

    def __init__(self, function: FunctionWithDerivatives, lr: float = 0.01, max_iters: int = 1000, tol: float = 1e-6,
                 verbose: bool = True):
        """
        Initialize the optimizer.

        :param function: Instance of FunctionWithDerivatives (with gradient method).
        :param lr: The step size for each update.
        :param max_iters: Maximum number of iterations for the optimization.
        :param tol: The stopping criterion based on the magnitude of the gradient.
        :param verbose: If True, prints detailed information during optimization.
        """
        self.function = function
        self.learning_rate = lr
        self.max_iters = max_iters
        self.tolerance = tol
        self.history = {
            'values': [],  # Stores the points visited during optimization
            'func_values': [],  # Stores the function values at each step
            'grad_magnitudes': []  # Stores the gradient magnitudes
        }
        self.verbose = verbose

    @abstractmethod
    def optimize(self, initial_values: np.ndarray) -> OptimizationResult:
        """
        Perform optimization. This method must be implemented by subclasses.
        :param initial_values: a NumPy array of initial values for the optimization variables
        :return: an OptimizationResult containing the optimized values, number of iterations, and elapsed time
        """
        pass

    def plot_convergence_trajectory_in_3d(self, x_range=(-2, 2), y_range=(-2, 2), resolution=100, show_arrows=True,
                                          arrow_every=1):
        """
        Plot the trajectory of the optimization on the 3D surface plot of the function.

        :param x_range: Tuple for x-axis range
        :param y_range: Tuple for y-axis range
        :param resolution: Number of points for meshgrid
        :param show_arrows: If True, plot step arrows along the trajectory
        :param arrow_every: Plot arrows every N steps to reduce clutter
        """
        trajectory = np.array(self.history['values'])
        x_vals, y_vals = trajectory[:, 0], trajectory[:, 1]
        z_vals = np.array([self.function.evaluate_function_at([x, y]) for x, y in trajectory])

        # Create the grid for surface plot
        x_vals_range = np.linspace(x_range[0], x_range[1], resolution)
        y_vals_range = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_vals_range, y_vals_range)

        Z = np.array([[self.function.evaluate_function_at([x, y])
                       for x, y in zip(x_row, y_row)]
                      for x_row, y_row in zip(X, Y)])

        # Create the 3D surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

        # Overlay the trajectory
        ax.plot(x_vals, y_vals, z_vals, 'ro-', label='Trajectory', linewidth=2)
        ax.scatter(x_vals, y_vals, z_vals, color='red', s=40)

        # Optionally add arrows between successive points
        if show_arrows:
            for i in range(0, len(trajectory) - 1, arrow_every):
                x0, y0, z0 = x_vals[i], y_vals[i], z_vals[i]
                dx, dy, dz = (x_vals[i + 1] - x0,
                              y_vals[i + 1] - y0,
                              z_vals[i + 1] - z0)

                ax.quiver(x0, y0, z0,
                          dx, dy, dz,
                          color='blue', arrow_length_ratio=0.3, linewidth=1)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        ax.set_title('Convergence Trajectory on 3D Surface')
        ax.legend()
        plt.show()

    def plot_convergence_trajectory_in_contour(self, x_range: tuple = (-2, 2), y_range: tuple = (-2, 2),
                                               resolution: int = 100, levels: int = 50) -> None:
        """
        Plot the trajectory of the optimization on the contour plot of the function.
        :param x_range: a tuple specifying the range of x values for the contour plot
        :param y_range: a tuple specifying the range of y values for the contour plot
        :param resolution: an integer specifying the resolution of the contour plot
        :param levels: an integer specifying the number of contour levels
        :return: a None, displays the contour plot with the trajectory
        """
        trajectory = np.array(self.history['values'])
        x_vals, y_vals = trajectory[:, 0], trajectory[:, 1]

        # Create the contour plot manually
        x_vals_range = np.linspace(x_range[0], x_range[1], resolution)
        y_vals_range = np.linspace(y_range[0], y_range[1], resolution)
        x_grid, y_grid = np.meshgrid(x_vals_range, y_vals_range)

        z_grid = np.array(
            [[self.function.evaluate_function_at([x, y]) for x, y in zip(x_row, y_row)] for x_row, y_row in
             zip(x_grid, y_grid)])

        # Plot the contour
        plt.contour(x_grid, y_grid, z_grid, levels=levels, cmap='viridis')
        plt.colorbar()

        # Plot the trajectory on top of the contour plot
        plt.plot(x_vals, y_vals, 'ro-', label='Trajectory')
        plt.scatter(x_vals, y_vals, color='red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Convergence Trajectory')
        plt.legend()
        plt.show()

    def plot_convergence_metrics(self) -> None:
        """
        Plot the function values and gradient magnitudes over iterations.
        """
        iterations = range(1, len(self.history['func_values']) + 1)

        # Plot function values
        plt.subplot(2, 1, 1)
        plt.plot(iterations, self.history['func_values'], 'b-', label='Function Value')
        plt.xlabel('Iteration')
        plt.ylabel('f(x)')
        plt.title('Function Value over Iterations')
        plt.legend()

        # Plot gradient magnitudes
        plt.subplot(2, 1, 2)
        plt.plot(iterations, self.history['grad_magnitudes'], 'r-', label='Gradient Magnitude')
        plt.xlabel('Iteration')
        plt.ylabel('||Gradient||')
        plt.title('Gradient Magnitude over Iterations')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def log(self, message) -> None:
        """
        Log a message if verbose mode is enabled.
        :param message: a string message to log
        :return: a None
        """
        if self.verbose:
            print(message)
