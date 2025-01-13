import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from pyvis.network import Network
import streamlit.components.v1 as components

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



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

# Persist state with Streamlit's session state
if "hidden_layers" not in st.session_state:
    st.session_state.hidden_layers = []
if "activations" not in st.session_state:
    st.session_state.activations = []

input_size = st.sidebar.number_input("Input Layer Size", min_value=1, value=784, step=1)
output_size = st.sidebar.number_input("Output Layer Size", min_value=1, value=10, step=1)

# Button to add a new hidden layer
if st.sidebar.button("Add Hidden Layer"):
    st.session_state.hidden_layers.append(256)  # Default size
    st.session_state.activations.append("ReLU")  # Default activation function

# Display, edit, or delete existing hidden layers
for i, (layer_size, activation) in enumerate(zip(st.session_state.hidden_layers, st.session_state.activations)):
    with st.sidebar.expander(f"Hidden Layer {i+1} ({layer_size} nodes, {activation})"):
        new_size = st.number_input(f"Edit Size for Hidden Layer {i+1}", min_value=1, value=layer_size, step=1, key=f"size_{i}")
        new_activation = st.selectbox(
            f"Activation for Hidden Layer {i+1}",
            ["ReLU", "Sigmoid", "Tanh"],
            index=["ReLU", "Sigmoid", "Tanh"].index(activation),
            key=f"activation_{i}"
        )
        if st.button(f"Save Changes to Layer {i+1}", key=f"save_{i}"):
            st.session_state.hidden_layers[i] = new_size
            st.session_state.activations[i] = new_activation

        if st.button(f"Delete Layer {i+1}", key=f"delete_{i}"):
            st.session_state.hidden_layers.pop(i)
            st.session_state.activations.pop(i)
            break  # Prevent index issues after deletion

# Visualize model button
if st.sidebar.button("Build Model"):
    if st.session_state.hidden_layers:
        html_file = visualize_interactive_model(input_size, st.session_state.hidden_layers, output_size)
        with open(html_file, "r") as f:
            html_content = f.read()
        components.html(html_content, height=600, width=800)
    else:
        st.error("Please add at least one hidden layer.")

def evaluate_model(model, test_loader):
    st.header("Model Evaluation")

    # Collect predictions and true labels
    all_preds = []
    all_labels = []

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.shape[0], -1)  # Flatten images
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # Get predicted classes
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")

    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_loader.dataset.classes,
                yticklabels=test_loader.dataset.classes, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    st.pyplot(fig)

# Option to train the model
def train_model():
    st.header("Train the Model")

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model
    model = create_model(input_size, st.session_state.hidden_layers, output_size, st.session_state.activations)
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

    # Save model in session state
    st.session_state.trained_model = model

    # Add evaluation button
    if st.sidebar.button("Evaluate Model"):
        evaluate_model(model, test_loader)


if st.sidebar.button("Train Model"):
    train_model()
