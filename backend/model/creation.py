import torch.nn as nn
import torchvision.models as models

def create_model(input_size=None, hidden_layers=None, output_size=None, activations=None, pretrained_model_name=None):
    """
    Creates a fully connected neural network or loads a pretrained model.

    Args:
        input_size (int): Number of input features (ignored if pretrained_model_name is provided).
        hidden_layers (list[int]): List of sizes for hidden layers (ignored if pretrained_model_name is provided).
        output_size (int): Number of output features (ignored if pretrained_model_name is provided).
        activations (list[str]): Activation functions for each hidden layer.
        pretrained_model_name (str): Name of a pretrained model to load. 

    Returns:
        nn.Module: The constructed or pretrained neural network model.
    """

    if pretrained_model_name:
        # Dictionary of available pretrained models
        pretrained_models = {
            'resnet18': models.resnet18(pretrained=True),
            'mobilenet_v2': models.mobilenet_v2(pretrained=True),
        }
        
        if pretrained_model_name in pretrained_models:
            print(f"Loading pretrained model: {pretrained_model_name}")
            return pretrained_models[pretrained_model_name]
        else:
            raise ValueError(f"Pretrained model '{pretrained_model_name}' is not available.")
    
    # Custom model creation
    if input_size is None or hidden_layers is None or output_size is None or activations is None:
        raise ValueError("For custom models, input_size, hidden_layers, output_size, and activations are required.")
    
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
