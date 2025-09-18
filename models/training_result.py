from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np


@dataclass
class TrainingResult:
    """
    A dataclass to store the result of a training process.
    Attributes:
        weights (Dict[str, np.ndarray]): The optimized weights and biases after training.
        epochs (int): The number of epochs the model was trained for.
        elapsed_time (float): The time taken to complete the training in seconds.
        loss_history (Optional[List[float]]): Optional list of scalar loss values per checkpoint.
    """
    weights: Dict[str, np.ndarray]
    epochs: int
    elapsed_time: Optional[float] = None
    loss_history: Optional[List[float]] = field(default_factory=list)
