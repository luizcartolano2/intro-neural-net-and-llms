class Activation:
    """
    Encapsulates an activation function and its derivative.
    Attributes:
        func: The activation function.
        derivative: The derivative of the activation function.
    """
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative
