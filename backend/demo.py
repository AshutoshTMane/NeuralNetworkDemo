import numpy as np

# Step 1: Define the structure of the neural network
# This example is a simple feedforward neural network with:
# - 1 input layer with 3 neurons
# - 1 hidden layer with 4 neurons
# - 1 output layer with 2 neurons

# A function to initialize weights and biases
# Weights are the connections between neurons, and biases help adjust the output
# during training.
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)  # For reproducibility
    
    # Random initialization of weights for input-to-hidden and hidden-to-output layers
    W1 = np.random.randn(hidden_size, input_size) * 0.01  # (hidden_size x input_size)
    W2 = np.random.randn(output_size, hidden_size) * 0.01  # (output_size x hidden_size)
    
    # Initialize biases as zeros
    b1 = np.zeros((hidden_size, 1))  # (hidden_size x 1)
    b2 = np.zeros((output_size, 1))  # (output_size x 1)
    
    return W1, b1, W2, b2  

# Step 2: Define the activation function
# Activation functions introduce non-linearity into the network.
# We'll use the sigmoid function for simplicity.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))