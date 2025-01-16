import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from model.creation import create_model
from model.evaluation import evaluate_model

from data.mnist_handwritting.mnist_handwritting import mnist_data


# Option to train the model
def train_model(input_size, output_size, epochs):
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
    timer_placeholder = st.empty()

    #animation_path = "Robot_Learning.gif" 
    #st.video(animation_path)

    #st.image("Robot_Learning.gif")

    # Timer variables
    total_batches = epochs * len(train_loader)
    batch_count = 0
    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            batch_count += 1

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

            # Progress and time estimation
            elapsed_time = time.time() - start_time
            time_per_batch = elapsed_time / batch_count
            remaining_batches = total_batches - batch_count
            estimated_time_remaining = remaining_batches * time_per_batch

            # Update progress and timer in Streamlit
            progress.progress(int(batch_count / total_batches * 100))
            timer_placeholder.write(f"Estimated time remaining: {estimated_time_remaining // 60:.0f}m {estimated_time_remaining % 60:.0f}s")

        losses.append(running_loss / len(train_loader))
            
        #st.write(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    plt.plot(range(1, epochs + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    st.pyplot(plt)

    # Save model in session state
    st.session_state.trained_model = model

    return model, test_loader