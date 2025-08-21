import numpy as np

from trainingRules.perceptron_rule import PerceptronRule


def f_net(f_in):
    """
    Activation function for the perceptron.
    :param f_in: a float representing the net input to the perceptron.
    :return: a binary output (1 or 0) based on the net input.
    """
    return 1 if f_in > 0 else 0


if __name__ == "__main__":
    # Input and Desired Output
    x = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0]])
    d = np.array([1, 1, 1, 0])
    # Initial Weights
    w = np.array([0, 0, 0])
    # Learning Rate
    eta = 0.5
    # Maximum Number of Iterations
    num_iterations = 100
    # Print Output
    print("=" * 90)
    print("Perceptron Rule:")
    perceptron_rule = PerceptronRule(eta, num_iterations, verbose=True)
    perceptron_results = perceptron_rule.fit(x, d, w.copy(), f_net)
    print(f"Final weights with Perceptron Rule: {perceptron_results.weights}")
    print("Number of epochs:", perceptron_results.epochs)
    print("Time taken for training: {:.6f} seconds".format(perceptron_results.elapsed_time))
    print("=" * 90)
