import torch
import torch.nn as nn
import os

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        # Define a simple feedforward neural network (ANN) for CIFAR-10 classification
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def save(self, filepath):
        """Save the model state dictionary to a file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath, device='cpu'):
        """Load model state dictionary from a file"""
        self.load_state_dict(torch.load(filepath, map_location=device))
        self.to(device)
        self.eval()  # Set to evaluation mode by default
        print(f"Model loaded from {filepath}")
        return self
    
    @classmethod
    def from_pretrained(cls, filepath, device='cpu'):
        """Create a model instance and load pretrained weights"""
        model = cls()
        model.load(filepath, device)
        return model