import time

import numpy as np

from models.training_result import TrainingResult


class PerceptronRule:
    """
    Implements the Perceptron learning rule.
    This rule updates the weights based on the error between the predicted and actual output.
    It iterates through the input data, calculates the predicted output, computes the error,
    and updates the weights accordingly.

    Attributes:
        lr (float): Learning rate for weight updates.
        max_iters (int): Maximum number of iterations to perform.
        verbose (bool): If True, prints detailed logs of the training process.

    Methods:
        fit(input_data: np.ndarray, output_data: np.ndarray, weights: np.ndarray, y_func: callable) -> np.ndarray:
            Trains the perceptron using the provided input and output data.
            Returns the updated weights after training.
    """
    def __init__(self, lr: float, max_iters: int, verbose: bool = False):
        self.lr = lr
        self.max_iters = max_iters
        self.verbose = verbose

    def __log(self, message: str = "") -> None:
        """
        Logs a message if verbose mode is enabled.
        :param message: a string message to log
        :return: a None
        """
        if self.verbose:
            print(message)

    def fit(self, input_data: np.ndarray, output_data: np.ndarray, weights: np.ndarray, y_func: callable) -> TrainingResult:
        """
        Trains the perceptron using the provided input and output data.
        :param input_data: a 2D NumPy array where each row is an input vector.
        :param output_data: a 1D NumPy array containing the expected outputs corresponding to the input vectors.
        :param weights: a 1D NumPy array representing the initial weights of the perceptron.
        :param y_func: a callable function that applies the activation function to the net input.
        :return: a 1D NumPy array of updated weights after training.
        """
        iter_i = 0
        has_weight_updated = True
        start_time = time.time()

        while iter_i < self.max_iters and has_weight_updated:
            self.__log("=" * 90)
            self.__log(f"Iteration {iter_i + 1}")
            self.__log("=" * 90)
            iter_j = 0
            has_weight_updated = False
            for input_i, output_i in zip(input_data, output_data):
                self.__log(f"\t[Sample {iter_j + 1}] Input: {input_i}, Expected: {output_i}")

                predict_y = np.dot(input_i, weights)
                self.__log(f"\t\tNet input (v): {predict_y:.4f} = {input_i} · {weights}")

                f_net_y = y_func(predict_y)
                self.__log(f"\t\tActivation f(v): {f_net_y:.4f}")

                error = output_i - f_net_y
                self.__log(f"\t\tError: {output_i} - {f_net_y:.4f} = {error:.4f}")

                if error != 0:
                    w_new = weights + self.lr * error * input_i
                    self.__log(f"\t\tWeight update: {weights} + {self.lr} * {error:.4f} * {input_i} = {w_new}")
                    weights = w_new
                    has_weight_updated = True
                else:
                    self.__log("\t\tNo weight update (prediction correct)")
                iter_j += 1
            iter_i += 1

        self.__log("=" * 90)

        return TrainingResult(
            weights=weights,
            epochs=iter_i,
            elapsed_time=time.time() - start_time
        )
