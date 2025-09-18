import time

import numpy as np

from models.training_result import TrainingResult


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, hidden_activation, output_activation):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_act = hidden_activation
        self.output_act = output_activation

        # Weights & biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

        self.predicted_output = None
        self.output_linear = None
        self.hidden_output = None
        self.hidden_linear = None

    def feedforward(self, x_input):
        self.hidden_linear = np.dot(x_input, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.hidden_act.func(self.hidden_linear)

        self.output_linear = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.output_act.func(self.output_linear)

        return self.predicted_output

    def backward(self, x_input, y, learning_rate):
        # Output layer
        output_error = y - self.predicted_output
        output_delta = output_error * self.output_act.derivative(self.predicted_output)

        # Hidden layer
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.hidden_act.derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(x_input.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate, log_interval=1000):
        start_time = time.time()
        loss_history = []

        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backward(X, y, learning_rate)

            if epoch % log_interval == 0:
                loss = np.sqrt(np.mean(np.square(y - output)))
                loss_history.append(loss)
                print(f"Epoch {epoch}, Loss: {loss}")

        elapsed_time = time.time() - start_time

        weights_dict = {
            "weights_input_hidden": self.weights_input_hidden.copy(),
            "weights_hidden_output": self.weights_hidden_output.copy(),
            "bias_hidden": self.bias_hidden.copy(),
            "bias_output": self.bias_output.copy()
        }

        return TrainingResult(
            weights=weights_dict,
            epochs=epochs,
            elapsed_time=elapsed_time,
            loss_history=loss_history
        )
