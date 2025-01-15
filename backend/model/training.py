import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from model.creation import create_model
from model.evaluation import evaluate_model

# Option to train the model
def train_model(input_size, output_size):
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
    #st.text(model)

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