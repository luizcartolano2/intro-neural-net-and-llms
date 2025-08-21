""" Project 1: Optimization of a Function with Derivatives
This script demonstrates the optimization of a function using The Steepest Descent and Gradient Descent methods.

It includes plotting the function's surface and contour, optimizing the function, and visualizing the convergence trajectory.
"""
import numpy as np
import sympy as sp
from numpy import e

from models.function_with_derivatives import FunctionWithDerivatives
from optimizers.gradient_descent import GradientDescent
from optimizers.steepest_descent import SteepestDescent

if __name__ == "__main__":
    # Initial Guess
    x_0 = np.array([0.3, 1.2])

    # Define the function using sympy
    x, y = sp.symbols('x y')
    func_expr = x * y * e ** (-x ** 2 - y ** 2)
    func = FunctionWithDerivatives((x, y), func_expr)

    # a. Plot the function at the initial guess
    func.plot_3d_surface(
        x_range=(-1, 1),
        y_range=(-1, 1),
        resolution=200,
        initial_point=x_0,
        show_gradient=False
    )
    func.plot_contour(
        x_range=(-1, 1),
        y_range=(-2, 2),
        resolution=200,
        levels=50,
        initial_point=x_0,
        show_gradient=False
    )

    # b. Plot the function at the initial guess with gradient
    func.plot_3d_surface(
        x_range=(-2, 2),
        y_range=(-1, 1),
        resolution=200,
        initial_point=x_0,
        show_gradient=True
    )
    func.plot_contour(
        x_range=(-1, 1),
        y_range=(-2, 2),
        resolution=200,
        levels=50,
        initial_point=x_0,
        show_gradient=True
    )

    # c. Optimize values
    # Control variables
    tolerance = 0.00001
    lr = 0.1

    steepest_descent_optimizer = SteepestDescent(function=func, max_iters=1000, lr=lr, tol=tolerance, verbose=False)
    steepest_descent_result = steepest_descent_optimizer.optimize(x_0)
    print("-" * 75)
    print("|Steepest Descent Result:")
    print(f"|\tOptimized values: {steepest_descent_result.x}")
    print(f"|\tFunction value at optimized point: {func.evaluate_function_at(steepest_descent_result.x)}")
    print(f"|\tNumber of iterations to converge: {steepest_descent_result.iterations}")
    print(f"|\tTime taken to converge: {steepest_descent_result.elapsed_time:.4f} seconds")
    print("-" * 75)

    gradient_descent_optimizer = GradientDescent(function=func, max_iters=1000, lr=lr, tol=tolerance, verbose=False)
    gradient_descent_result = gradient_descent_optimizer.optimize(x_0)
    print("-" * 75)
    print("|Gradient Descent Result:")
    print(f"|\tOptimized values: {gradient_descent_result.x}")
    print(f"|\tFunction value at optimized point: {func.evaluate_function_at(gradient_descent_result.x)}")
    print(f"|\tNumber of iterations to converge: {gradient_descent_result.iterations}")
    print(f"|\tTime taken to converge: {gradient_descent_result.elapsed_time:.4f} seconds")
    print("-" * 75)

    # d. Plot function with the min point found
    # The Steepest Descent
    func.plot_3d_surface(
        x_range=(-1, 0),
        y_range=(0, 1),
        resolution=200,
        initial_point=steepest_descent_result.x,
        show_gradient=False
    )
    func.plot_contour(
        x_range=(-1, 1),
        y_range=(-2, 2),
        resolution=200,
        levels=50,
        initial_point=steepest_descent_result.x,
        show_gradient=True
    )
    # Gradient Descent
    func.plot_3d_surface(
        x_range=(-1, 0),
        y_range=(0, 1),
        resolution=200,
        initial_point=gradient_descent_result.x,
        show_gradient=False
    )
    func.plot_contour(
        x_range=(-1, 1),
        y_range=(-2, 2),
        resolution=200,
        levels=50,
        initial_point=gradient_descent_result.x,
        show_gradient=True
    )

    # e. Plot convergence trajectory
    # The Steepest Descent
    steepest_descent_optimizer.plot_convergence_trajectory_in_3d(
        x_range=(-1, 1),
        y_range=(-2, 2),
        resolution=200
    )
    steepest_descent_optimizer.plot_convergence_trajectory_in_contour(
        x_range=(-1, 1),
        y_range=(-2, 2),
        resolution=200,
        levels=50
    )
    # Gradient Descent
    gradient_descent_optimizer.plot_convergence_trajectory_in_3d(
        x_range=(-1, 1),
        y_range=(-2, 2),
        resolution=200
    )
    gradient_descent_optimizer.plot_convergence_trajectory_in_contour(
        x_range=(-1, 1),
        y_range=(-2, 2),
        resolution=200,
        levels=50
    )
