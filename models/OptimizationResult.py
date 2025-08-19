from dataclasses import dataclass
import numpy as np


@dataclass
class OptimizationResult:
    """
    A named tuple to store the result of an optimization process.
    Attributes:
        x (np.ndarray): The optimized values of the variables.
        iterations (int): The number of iterations taken to converge.
        elapsed_time (float): The time taken to complete the optimization in seconds.
    """
    x: np.ndarray
    iterations: int
    elapsed_time: float
