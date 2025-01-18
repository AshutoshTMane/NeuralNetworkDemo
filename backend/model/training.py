import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from model.creation import create_model

from data.mnist_handwritting.mnist_handwritting import mnist_data


# Option to train the model
def train_model(model, epochs, learning_rate=0.001):
    """
    Trains a given model with the specified parameters.

    Args:
        model (nn.Module): The neural network model to train.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    """
    st.header("Train the Model")

    # Load data
    train_loader, test_loader = mnist_data()

    # Log model structure and parameters
    st.write("Model Structure:")
    st.text(model)
    st.write(f"Training for {epochs} epochs with learning rate {learning_rate:.4f}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Variables for progress tracking
    losses = []
    progress = st.progress(0)
    timer_placeholder = st.empty()

    total_batches = epochs * len(train_loader)
    batch_count = 0
    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            batch_count += 1

            # Training logic
            progress.progress(int(batch_count / total_batches * 100))

            # Move data to device
            images = images.view(images.shape[0], -1).to(device)  # Flatten images
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Progress and time estimation
            elapsed_time = time.time() - start_time
            time_per_batch = elapsed_time / batch_count
            remaining_batches = total_batches - batch_count
            estimated_time_remaining = remaining_batches * time_per_batch

            # Update timer display
            timer_placeholder.write(f"Estimated time remaining: {estimated_time_remaining // 60:.0f}m "
                                    f"{estimated_time_remaining % 60:.0f}s")

        # Log epoch loss
        losses.append(running_loss / len(train_loader))
        st.write(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Plot loss curve
    plt.plot(range(1, epochs + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    st.pyplot(plt)

    # Save the trained model in session state
    st.session_state.trained_model = model
    st.success("Model training complete!")

    return model, test_loader
