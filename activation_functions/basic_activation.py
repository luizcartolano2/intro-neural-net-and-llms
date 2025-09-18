import numpy as np

from activation_functions.activation import Activation
from typing import Dict


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    :param x: any input array
    :return: an array with the sigmoid function applied element-wise
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the sigmoid function. Assumes x is already sigmoid(x).
    :param x: an array where the sigmoid function has been applied
    :return: an array with the derivative of the sigmoid function applied element-wise
    """
    return x * (1 - x)


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation function.
    :param x: any input array
    :return: an array with the ReLU function applied element-wise
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU. Assumes x is already relu(x).
    :param x: an array where the ReLU function has been applied
    :return: an array with the derivative of the ReLU function applied element-wise
    """
    return (x > 0).astype(float)


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Tanh activation function.
    :param x: any input array
    :return: an array with the tanh function applied element-wise
    """
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh. Assumes x is already tanh(x).
    :param x: an array where the tanh function has been applied
    :return: an array with the derivative of the tanh function applied element-wise
    """
    return 1 - x ** 2


# Mapping activation names to their functions
_ACTIVATIONS: Dict[str, Activation] = {
    "sigmoid": Activation(sigmoid, sigmoid_derivative),
    "relu": Activation(relu, relu_derivative),
    "tanh": Activation(tanh, tanh_derivative),
}


def get_activation_function(name: str) -> Activation:
    """
    Retrieve an activation function by name.
    :param name: a string name of the activation function ("sigmoid", "relu", "tanh")
    :return: an Activation object containing the function and its derivative
    :raises ValueError: if the activation function name is unknown
    """
    try:
        return _ACTIVATIONS[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown activation function: {name}")
