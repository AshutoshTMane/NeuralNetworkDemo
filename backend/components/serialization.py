import torch
import streamlit as st

def save_model_to_file(model, path="trained_model.pth"):
    """
    Saves the given PyTorch model to a file.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The path where the model will be saved.
    """
    try:
        torch.save(model.state_dict(), path)
        return f"Model saved to {path}"
    except Exception as e:
        return f"Error saving model: {e}"


def load_model_from_file(model, path="trained_model.pth"):
    """
    Loads the model state from a file into the given model architecture.

    Args:
        model (torch.nn.Module): The model instance to load the state into.
        path (str): The path from where the model state will be loaded.

    Returns:
        tuple: A tuple containing the loaded model and a message indicating success or failure.
    """
    try:
        model.load_state_dict(torch.load(path))
        model.eval()  # Ensure the model is in evaluation mode
        return model, f"Model loaded successfully from {path}"
    except FileNotFoundError:
        return None, "No saved model file found. Please save a model first."
    except Exception as e:
        return None, f"Error loading model: {e}"


def handle_model_saving(model):
    """
    Streamlit wrapper for saving the model with user interaction.

    Args:
        model (torch.nn.Module): The model to save.
    """
    if st.button("Save Model"):
        if model:
            message = save_model_to_file(model)
            st.success(message)
        else:
            st.error("No model available to save. Train or load a model first.")


def handle_model_loading(model_architecture):
    """
    Streamlit wrapper for loading the model with user interaction.

    Args:
        model_architecture (torch.nn.Module): An uninitialized model architecture.

    Returns:
        torch.nn.Module: The loaded model.
    """
    if st.button("Load Model"):
        model, message = load_model_from_file(model_architecture)
        if model:
            st.session_state['trained_model'] = model
            st.success(message)
        else:
            st.error(message)
