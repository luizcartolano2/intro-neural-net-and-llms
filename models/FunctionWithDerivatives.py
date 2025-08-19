import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


class FunctionWithDerivatives:
    """
    A class to represent a mathematical function with its derivatives,
    including the gradient and Hessian matrix.
    This class allows for evaluation of the function, its gradient,
    and Hessian at specific points, as well as plotting the function's
    3D surface and contour plots.

    Attributes:
        symbols (tuple): A tuple of sympy symbols representing the variables of the function.
        func_expr (sympy expression): The sympy expression representing the function.
        grad_expr (list): The gradient of the function as a list of sympy expressions.
        hessian_expr (sympy Matrix): The Hessian matrix of the function.

    Methods:
        evaluate_function_at(values): Evaluates the function at given values.
        evaluate_gradient_at(values): Evaluates the gradient at given values.
        evaluate_hessian_at(values): Evaluates the Hessian matrix at given values.
        plot_3d_surface(x_range, y_range, resolution): Plots the 3D surface of the function.
        plot_contour(x_range, y_range, resolution, levels): Plots the contour of the function.
    """

    def __init__(self, symbols: tuple, func_expr: sp.Expr) -> None:
        """
        Initialize the class with the symbols and the function expression.

        :param symbols: A tuple of sympy symbols (e.g., (x1, x2))
        :param func_expr: A sympy expression for the function
        """
        self.symbols = symbols
        self.func_expr = func_expr
        self.grad_expr = self.__get_gradient()
        self.hessian_expr = self.__get_hessian()

    def evaluate_function_at(self, values: list or np.ndarray) -> float:
        """
        Evaluate the function at given values.
        :param values: a list or array of values corresponding to the symbols
        :return: a float value of the function evaluated at the given values
        """
        values_dict = {symbol: value for symbol, value in zip(self.symbols, values)}
        func_value = float(self.func_expr.subs(values_dict))

        return func_value

    def __get_gradient(self) -> list:
        """
        Compute the gradient of the function.
        :return: a list of sympy expressions representing the gradient
        """
        gradient = [sp.diff(self.func_expr, symbol) for symbol in self.symbols]
        return gradient

    def evaluate_gradient_at(self, values: list or np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient at given values.
        :param values: a list or array of values corresponding to the symbols
        :return: a numpy array of gradient values evaluated at the given values
        """
        values_dict = {symbol: value for symbol, value in zip(self.symbols, values)}
        grad_values = np.array([float(grad.subs(values_dict)) for grad in self.grad_expr])

        return grad_values

    def __get_hessian(self) -> sp.Matrix:
        """
        Compute the Hessian matrix of the function.
        :return: a sympy Matrix representing the Hessian
        """
        hessian = sp.Matrix([[sp.diff(self.func_expr, s1, s2) for s2 in self.symbols] for s1 in self.symbols])
        return hessian

    def evaluate_hessian_at(self, values: list or np.ndarray) -> np.ndarray:
        """
        Evaluate the Hessian matrix at given values.
        :param values: a list or array of values corresponding to the symbols
        :return: a numpy array of Hessian values evaluated at the given values
        """
        values_dict = {symbol: value for symbol, value in zip(self.symbols, values)}
        hessian_matrix = self.hessian_expr.subs(values_dict)
        hessian_values = np.array(hessian_matrix.tolist()).astype(np.float64)

        return hessian_values

    def plot_3d_surface(self, x_range: tuple = (-2, 2), y_range: tuple = (-2, 2), resolution: int = 100,
                        initial_point: np.ndarray = None, show_gradient: bool = False) -> None:
        """
        Plot the 3D surface of the function over a specified range.

        :param x_range: Tuple specifying the range for the x-axis (e.g., (-2, 2))
        :param y_range: Tuple specifying the range for the y-axis (e.g., (-2, 2))
        :param resolution: Number of points for meshgrid (e.g., 100)
        :param initial_point: Optional initial point to highlight on the surface
        :param show_gradient: If True, shows the gradient at the initial point
        """
        if show_gradient:
            assert initial_point is not None, "Initial point must be provided to show gradient."

        x_vals = np.linspace(*x_range, resolution)
        y_vals = np.linspace(*y_range, resolution)
        x_rows, y_rows = np.meshgrid(x_vals, y_vals)

        # Evaluate the function on the grid
        z_row = np.array(
            [[self.evaluate_function_at([x, y]) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(x_rows, y_rows)])

        # Create the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_rows, y_rows, z_row, cmap='viridis', edgecolor='none')

        if initial_point is not None:
            x_0, y_0 = initial_point
            z_0 = self.evaluate_function_at([x_0, y_0])
            ax.scatter(x_0, y_0, z_0, color='red', s=resolution, label='Initial Point')

            if show_gradient:
                grad = self.evaluate_gradient_at([x_0, y_0])
                ax.quiver(x_0, y_0, z_0, -grad[0], -grad[1], 0, length=0.5, color='blue', normalize=True,
                          label='Gradient')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        ax.set_title('3D Surface Plot')
        plt.show()

    def plot_contour(self, x_range: tuple = (-2, 2), y_range: tuple = (-2, 2), resolution: int = 100,
                     levels: int = 50, initial_point: np.ndarray = None, show_gradient: bool = False) -> None:
        """
        Plot the contour of the function over a specified range.

        :param x_range: Tuple specifying the range for the x-axis (e.g., (-2, 2))
        :param y_range: Tuple specifying the range for the y-axis (e.g., (-2, 2))
        :param resolution: Number of points for meshgrid (e.g., 100)
        :param levels: Number of contour levels
        :param initial_point: Optional initial point to highlight on the surface
        :param show_gradient: If True, shows the gradient at the initial point
        """
        if show_gradient:
            assert initial_point is not None, "Initial point must be provided to show gradient."

        x_vals = np.linspace(*x_range, resolution)
        y_vals = np.linspace(*y_range, resolution)
        x_rows, y_rows = np.meshgrid(x_vals, y_vals)

        # Evaluate the function on the grid
        z_rows = np.array(
            [[self.evaluate_function_at([x, y]) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(x_rows, y_rows)])

        # Create the contour plot
        plt.contour(x_rows, y_rows, z_rows, levels=levels, cmap='viridis')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Contour Plot')

        if initial_point is not None:
            x_0, y_0 = initial_point
            plt.scatter(x_0, y_0, color='red', s=100, label='Initial Point')

            if show_gradient:
                grad = self.evaluate_gradient_at([x_0, y_0])
                plt.quiver(x_0, y_0, -grad[0], -grad[1],
                           angles='xy', scale_units='xy', scale=1,
                           color='blue', label='Gradient')

        plt.show()
