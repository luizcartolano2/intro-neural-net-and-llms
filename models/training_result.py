from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingResult:
    """
    A named tuple to store the result of a training process.
    Attributes:
        weights (np.ndarray): The optimized weights of the model after training.
        epochs (int): The number of epochs the model was trained for.
        elapsed_time (float): The time taken to complete the training in seconds.
    """
    weights: np.ndarray
    epochs: int
    elapsed_time: float
