import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activations):
        """
        Initialize a customizable neural network.
        
        Args:
            input_size (int): Number of input features.
            hidden_layers (list of int): Number of neurons in each hidden layer.
            output_size (int): Number of output features.
            activations (list of nn.Module): Activation functions for each layer.
        """
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Define the layers dynamically
        layers = []
        prev_size = input_size
        
        for i, (hidden_size, activation) in enumerate(zip(hidden_layers, activations)):
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation:
                layers.append(activation())
            prev_size = hidden_size
        
        # Add the final output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.linear_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

def get_user_input():
    """
    Collect neural network parameters from the user interactively.
    """
    print("Define your custom neural network:")

    # Input size
    input_size = int(input("Enter the input size (e.g., 784 for 28x28 images): "))

    # Hidden layers
    hidden_layers = input("Enter the sizes of hidden layers as a comma-separated list (e.g., 512,256,128): ")
    hidden_layers = [int(size.strip()) for size in hidden_layers.split(",")]

    # Output size
    output_size = int(input("Enter the output size (e.g., 10 for a 10-class classification problem): "))

    # Activation functions
    print("Choose activation functions for each layer:")
    activations = []
    for i in range(len(hidden_layers)):
        print(f"Hidden layer {i + 1}:")
        print("1. ReLU")
        print("2. Sigmoid")
        print("3. Tanh")
        choice = int(input("Select an activation function (1, 2, or 3): "))
        if choice == 1:
            activations.append(nn.ReLU)
        elif choice == 2:
            activations.append(nn.Sigmoid)
        elif choice == 3:
            activations.append(nn.Tanh)
        else:
            print("Invalid choice, defaulting to ReLU.")
            activations.append(nn.ReLU)

    return input_size, hidden_layers, output_size, activations

# Collect user input
input_size, hidden_layers, output_size, activations = get_user_input()

# Create the neural network
model = NeuralNetwork(input_size, hidden_layers, output_size, activations)

# Display the model structure
print("\nYour customized neural network:")
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')