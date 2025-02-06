import streamlit as st
import torch.nn as nn
import torchvision.models as models

def create_model(input_size=None, hidden_layers=None, output_size=None, activations=None):
    """
    Creates a fully connected neural network.

    Args:
        input_size (int): Number of input features.
        hidden_layers (list[int]): List of sizes for hidden layers.
        output_size (int): Number of output features.
        activations (list[str]): Activation functions for each hidden layer.

    Returns:
        nn.Module: The constructed neural network model.
    """
    
    # Ensure all required parameters are provided
    if input_size is None or hidden_layers is None or output_size is None or activations is None:
        raise ValueError("For custom models, input_size, hidden_layers, output_size, and activations are required.")
    
    layers = []  # List to store layers of the model
    in_features = input_size  # Initial number of input features

    # Iterate through the hidden layers and add them to the model
    for i, out_features in enumerate(hidden_layers):
        layers.append(nn.Linear(in_features, out_features))  # Add a linear layer
        activation = getattr(nn, activations[i], None)  # Get the activation function
        if activation is not None:
            layers.append(activation())  # Add activation function if valid
        in_features = out_features  # Update input size for the next layer

    # Add the final output layer
    layers.append(nn.Linear(in_features, output_size))
    
    # Update Streamlit session state to indicate a model has been created
    st.session_state["model_selected"] = True

    # Return the constructed model as a sequential container of layers
    return nn.Sequential(*layers)
