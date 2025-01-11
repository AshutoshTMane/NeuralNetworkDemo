import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from pyvis.network import Network
import os
import streamlit.components.v1 as components

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

# Function to visualize the model using Pyvis
def visualize_interactive_model(input_size, hidden_layers, output_size):
    net = Network(height="500px", width="800px", directed=True)

    net.add_node("Input", label="Input ({} nodes)".format(input_size), color="#f4a261", shape="circle")
    previous_layer = "Input"

    for i, layer_size in enumerate(hidden_layers):
        layer_name = f"Hidden Layer {i+1}"
        net.add_node(layer_name, label=f"Hidden Layer {i+1} ({layer_size} nodes)", color="#2a9d8f", shape="circle")
        net.add_edge(previous_layer, layer_name)
        previous_layer = layer_name

    net.add_node("Output", label="Output ({} nodes)".format(output_size), color="#e76f51", shape="circle")
    net.add_edge(previous_layer, "Output")

    net.set_options("""
    var options = {
      "physics": {
        "enabled": true
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed"
        }
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    html_file = "interactive_model.html"
    net.save_graph(html_file)
    return html_file

# Sidebar for user input
st.sidebar.header("Neural Network Configuration")
input_size = st.sidebar.number_input("Input Layer Size", min_value=1, value=784, step=1)
output_size = st.sidebar.number_input("Output Layer Size", min_value=1, value=10, step=1)

hidden_layers_input = st.sidebar.text_input(
    "Hidden Layers (comma-separated, e.g., 512,256,128)", value="512,256,128"
)
try:
    hidden_layers = [
        int(size.strip()) for size in hidden_layers_input.split(",") if int(size.strip()) > 0
    ]
    if not hidden_layers:
        raise ValueError("At least one hidden layer must be specified.")
except ValueError:
    st.sidebar.error("Invalid input! Enter positive integers separated by commas.")
    hidden_layers = []

activations = []
for i in range(len(hidden_layers)):
    activations.append(
        st.sidebar.selectbox(
            f"Activation for Hidden Layer {i+1}", ["ReLU", "Sigmoid", "Tanh"], index=0
        )
    )

# Visualize model button
if st.sidebar.button("Build Model"):
    if hidden_layers:
        html_file = visualize_interactive_model(input_size, hidden_layers, output_size)
        with open(html_file, "r") as f:
            html_content = f.read()
        components.html(html_content, height=600, width=800)
    else:
        st.error("Please configure the hidden layers correctly.")

# Option to train the model
def train_model():
    st.header("Train the Model")

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Model
    model = create_model(input_size, hidden_layers, output_size, activations)
    st.text(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = st.sidebar.slider("Epochs", min_value=1, max_value=20, value=5)
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        st.write(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

if st.sidebar.button("Train Model"):
    train_model()
