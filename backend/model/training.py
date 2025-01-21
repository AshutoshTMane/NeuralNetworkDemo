import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, dataset, epochs, learning_rate, batch_size=32):
    """
    Trains a given model with the specified dataset and parameters.

    Args:
        model (nn.Module): The neural network model to train.
        dataset (tuple): A tuple containing the train and test data loaders.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    """
    st.header("Train the Model")

        # Initialize session state variables
    if "training_progress" not in st.session_state:
        st.session_state.training_progress = 0
    if "epoch_losses" not in st.session_state:
        st.session_state.epoch_losses = []
    if "current_epoch" not in st.session_state:
        st.session_state.current_epoch = 0

    # Check if dataset is a pandas DataFrame, assuming it's structured with features and labels
    if isinstance(dataset, pd.DataFrame):
        # Split the dataset into features (X) and labels (y)
        X = dataset.iloc[:, :-1].values  # All columns except the last one
        y = dataset.iloc[:, -1].values   # Last column as labels

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Create DataLoader instances
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    else:
        st.error("Unsupported dataset format")
        return

    # Log model structure and parameters
    #st.write("Model Structure:")
    #st.text(model)
    #st.write(f"Training for {epochs} epochs with learning rate {learning_rate:.4f}")

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

    epoch_progress = st.empty()

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
        epoch_loss = running_loss / len(train_loader)
        st.session_state.epoch_losses.append(epoch_loss)
        st.session_state.current_epoch += 1
        epoch_progress.write(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        

    # Plot loss curve
    plt.plot(range(1, len(st.session_state.epoch_losses) + 1), st.session_state.epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    st.pyplot(plt)

    # Save the trained model in session state
    st.session_state.trained_model = model
    st.session_state.test_loader = test_loader  # Save the test dataset in session state
    st.success("Model training complete!")

    return model, test_loader
