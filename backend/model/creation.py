import torch.nn as nn

# Function to dynamically create a neural network model
def create_model(input_size, hidden_layers, output_size, activations):
    """
    Creates a fully connected neural network where each layer uses the same activation function.

    Args:
        input_size (int): Number of input features.
        hidden_layers (list[int]): List of sizes for hidden layers.
        output_size (int): Number of output features.
        activation (str): Activation function to use for all hidden layers.

    Returns:
        nn.Sequential: The constructed neural network model.
    """
    layers = []
    in_features = input_size

    for i, out_features in enumerate(hidden_layers):
        layers.append(nn.Linear(in_features, out_features))
        activation = getattr(nn, activations[i], None)
        if activation is not None:
            layers.append(activation())
        in_features = out_features

    layers.append(nn.Linear(in_features, output_size))
    return nn.Sequential(*layers)
