import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        A simple 10-layer MLP:
            - 1 input layer
            - 8 hidden layers (with ReLU)
            - 1 output layer
        """
        super(MLP, self).__init__()

        layers = []
        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Next 8 - 2 = 6 more hidden layers (for a total of 8 hidden layers)
        # Or you can explicitly do 8 hidden layers if you want each enumerated.
        for _ in range(7):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Final output layer: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Combine all layers into a Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    # Hyperparameters
    input_size = 784  # Example: 28x28 flattened image
    hidden_size = 256  # Example hidden size
    output_size = 10  # Example: number of classes (e.g., MNIST)
    batch_size = 32

    # Create a random tensor to represent a batch of data
    dummy_input = torch.randn(batch_size, input_size)

    # Instantiate the model
    model = MLP(input_size, hidden_size, output_size)

    # Forward pass through the model
    output = model(dummy_input)
    print("Output shape:", output.shape)  # should be [batch_size, output_size]

    # Define a sample loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Dummy target labels for the sake of demonstration
    target = torch.randint(0, output_size, (batch_size,))

    # Compute the loss
    loss = criterion(output, target)
    print("Initial loss:", loss.item())

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Model training step complete.")
