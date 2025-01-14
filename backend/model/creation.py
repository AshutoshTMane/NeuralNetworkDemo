

# Function to dynamically create a neural network model
def create_model(input_size, hidden_layers, output_size, activations):
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
