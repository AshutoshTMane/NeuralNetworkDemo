import torch

def save_model(model, path="model.pth"):
    """
    Save the PyTorch model to a file.

    Args:
        model (torch.nn.Module): Trained PyTorch model to save.
        path (str): Path to save the model file.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="model.pth"):
    """
    Load the PyTorch model from a file.

    Args:
        model (torch.nn.Module): Uninitialized PyTorch model with the same architecture as the saved model.
        path (str): Path to the saved model file.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")
    return model
