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

# Derivative of sigmoid for backpropagation
# This helps the network learn during training by updating weights and biases.
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Step 3: Forward propagation
# This calculates the output of the network based on inputs and current weights/biases.
def forward_propagation(X, W1, b1, W2, b2):
    # Linear step for the hidden layer
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)  # Activation for the hidden layer

    # Linear step for the output layer
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)  # Activation for the output layer

    return Z1, A1, Z2, A2

# Step 4: Compute the cost
# The cost function measures how well the network's predictions match the actual outputs.
def compute_cost(A2, Y):
    m = Y.shape[1]  # Number of examples
    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    return cost
