import numpy as np
from matplotlib import pyplot as plt

from activation_functions.basic_activation import get_activation_function
from models.simple_neural_network import SimpleNeuralNetwork

if __name__ == "__main__":

    SIGMOID = get_activation_function("sigmoid")
    nn = SimpleNeuralNetwork(
        input_size=2,
        hidden_size=2,
        output_size=2,
        hidden_activation=SIGMOID,
        output_activation=SIGMOID
    )

    X_train = np.array([
        [0.5, 0.2],  # Maçã
        [0.8, 0.6]  # Laranja
    ])

    y_train = np.array([
        [1, 0],  # Maçã
        [0, 1]  # Laranja
    ])

    # Train the network
    log_interval = 100
    result = nn.train(X_train, y_train, epochs=1000, learning_rate=0.5, log_interval=log_interval)
    print("Training completed in", result.elapsed_time, "seconds")
    print("Final Loss checkpoints:", result.loss_history)

    with open('results/project3/sigmoid_results', 'w') as f:
        f.write("Trained for epochs: " + str(result.epochs) + "\n")
        f.write("Elapsed time (s): " + str(result.elapsed_time) + "\n")
        f.write("Final loss: " + str(result.loss_history[-1]) + "\n")
        f.write("Input->Hidden weights:\n" + str(result.weights['weights_input_hidden']) + "\n")
        f.write("Hidden->Output weights:\n" + str(result.weights['weights_hidden_output']) + "\n")
        f.write("Bias Hidden:\n" + str(result.weights['bias_hidden']) + "\n")
        f.write("Bias Output:\n" + str(result.weights['bias_output']) + "\n")

    plt.figure(figsize=(8, 5))
    plt.plot([i * log_interval for i in range(len(result.loss_history))], result.loss_history, marker='o')
    # plt.title("Training Loss (every {} epochs)".format(log_interval))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig('results/project3/sigmoid_loss_plot.png')
