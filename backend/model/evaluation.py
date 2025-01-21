import streamlit as st
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms


def evaluate_model(model, test_loader):
    st.subheader("Model Evaluation")

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


def predict_single_sample(model, image, transform=None):
    st.subheader("Single Sample Prediction")

    # Preprocess the input
    if transform:
        image = transform(image).unsqueeze(0)  # Apply transform and add batch dimension
    else:
        image = torch.tensor(np.array(image)).float().unsqueeze(0)

    # Debug: Check input shape
    st.write("Input shape:", image.shape)

    # Resize or reshape the input to match the model's new input size
    image = F.adaptive_avg_pool2d(image, (1, 64)).view(1, -1)  # Resize to 1x64 vector

    # Predict using the model
    model.eval()
    with torch.no_grad():
        output = model(image)
        st.write("Model output:", output)  # Debug: Raw model output
        pred = torch.argmax(output, dim=1).item()

        # Assuming `output` is the raw model output
    probabilities = F.softmax(output, dim=1)
    predicted_digit = torch.argmax(probabilities, dim=1).item()

    print(f"Model output (logits): {output}")
    print(f"Probabilities: {probabilities}")
    print(f"Predicted Digit: {predicted_digit}")

    return pred
