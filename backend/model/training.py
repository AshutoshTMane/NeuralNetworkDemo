import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model.creation import create_model
from model.evaluation import evaluate_model

from data.mnist_handwritting.mnist_handwritting import mnist_data


# Option to train the model
def train_model(input_size, output_size):
    st.header("Train the Model")

    # Load data
    train_loader, test_loader = mnist_data()

    # Model
    model = create_model(input_size, st.session_state.hidden_layers, output_size, st.session_state.activations)
    #st.text(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []
    progress = st.progress(0)
    # Training loop
    epochs = st.sidebar.slider("Epochs", min_value=1, max_value=20, value=5)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Training logic
            progress.progress(int((epoch * len(train_loader) + i + 1) / (epochs * len(train_loader)) * 100))
            images = images.view(images.shape[0], -1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        losses.append(running_loss / len(train_loader))
            
        st.write(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    plt.plot(range(1, epochs + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    st.pyplot(plt)

    # Save model in session state
    st.session_state.trained_model = model

    # Add evaluation button
    if st.sidebar.button("Evaluate Model"):
        evaluate_model(model, test_loader)