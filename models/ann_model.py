import torch.nn as nn

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        # Define a simple feedforward neural network (ANN) for CIFAR-10 classification
        self.network = nn.Sequential(
            # Input: 3x32x32 (CIFAR-10 images)
            nn.Flatten(),
            nn.Linear(32*32*3, 512),
            # Hidden layer 1
            nn.ReLU(),
            nn.Linear(512,256),
            # Hidden layer 2
            nn.ReLU(),
            nn.Linear(256,10)
        )

    def forward(self, x):
        return self.network(x)