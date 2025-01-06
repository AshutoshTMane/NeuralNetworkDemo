import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import streamlit as st
from torchviz import make_dot
from PIL import Image
import io

# Streamlit app starts here
st.title("Neural Network Builder and Trainer")

# Sidebar inputs
st.sidebar.header("Network Configuration")

# Input size
input_size = st.sidebar.number_input(
    "Input Size", min_value=1, max_value=1024, value=28 * 28, step=1
)

# Hidden layers
hidden_layers_input = st.sidebar.text_input(
    "Hidden Layers (comma-separated, e.g., 512,256,128)", value="512,256,128"
)
try:
    hidden_layers = [int(size.strip()) for size in hidden_layers_input.split(",") if int(size.strip()) > 0]
except ValueError:
    st.sidebar.error("Invalid hidden layer sizes. Ensure all values are positive integers.")
    hidden_layers = []

# Output size
output_size = st.sidebar.number_input(
    "Output Size", min_value=1, max_value=1024, value=10, step=1
)

# Activation functions
activations = []
for i in range(len(hidden_layers)):
    activation = st.sidebar.selectbox(
        f"Activation Function for Layer {i + 1}",
        ["ReLU", "Sigmoid", "Tanh"],
        index=0,
    )
    activations.append(getattr(nn, activation))

# Model visualization function
def visualize_model(model):
    try:
        sample_input = torch.randn(1, input_size)
        model_graph = make_dot(model(sample_input), params=dict(model.named_parameters()))
        model_graph.render("model", format="png", cleanup=True)
        return Image.open("model.png")
    except Exception as e:
        st.error(f"Error visualizing model: {e}")
        return None

# Create the network dynamically
st.sidebar.header("Model")
if st.sidebar.button("Build Model"):
    model = nn.Sequential()
    prev_size = input_size

    # Adding layers dynamically
    for hidden_size, activation in zip(hidden_layers, activations):
        model.add_module(f"Linear_{prev_size}_{hidden_size}", nn.Linear(prev_size, hidden_size))
        model.add_module(f"Activation_{activation.__name__}", activation())
        prev_size = hidden_size

    model.add_module(f"Linear_{prev_size}_{output_size}", nn.Linear(prev_size, output_size))
    st.sidebar.success("Model built successfully!")

    # Visualize the model
    st.header("Neural Network Diagram")
    network_image = visualize_model(model)
    if network_image:
        st.image(network_image, caption="Network Diagram", use_column_width=True)

# Data Loading
st.sidebar.header("Dataset Configuration")
batch_size = st.sidebar.number_input("Batch Size", min_value=1, max_value=512, value=64, step=1)

if st.sidebar.button("Load Dataset"):
    try:
        training_data = datasets.FashionMNIST(
            root="data", train=True, download=True, transform=ToTensor()
        )
        test_data = datasets.FashionMNIST(
            root="data", train=False, download=True, transform=ToTensor()
        )
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        st.sidebar.success("Dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading dataset: {e}")

# Training
if "model" in locals() and "train_dataloader" in locals():
    st.sidebar.header("Training Configuration")
    epochs = st.sidebar.slider("Epochs", min_value=1, max_value=100, value=5, step=1)
    learning_rate = st.sidebar.slider("Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    st.header("Training Progress")
    progress_bar = st.progress(0)
    train_loss_chart = st.line_chart()
    accuracy_chart = st.line_chart()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for X, y in train_dataloader:
            X, y = X.view(X.size(0), -1), y  # Flatten input
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(train_dataloader)
        accuracy = correct / total
        st.write(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        train_loss_chart.add_rows({"Loss": [avg_loss]})
        accuracy_chart.add_rows({"Accuracy": [accuracy]})
        progress_bar.progress((epoch + 1) / epochs)

    st.sidebar.success("Training complete!")
