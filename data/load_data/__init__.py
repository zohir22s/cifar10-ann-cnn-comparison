# load_data.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Standard CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
    root="data/raw",
    train=True,
    download=True,
    transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="data/raw",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Noisy CIFAR-10 for testing robustness

# We'll add a little random noise to each image to simulate a "noisy" dataset.
noise_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.1*torch.randn_like(x)),  # small Gaussian noise
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Create training and testing datasets with noise
noisy_train_dataset = torchvision.datasets.CIFAR10(
    root="data/noisy",
    train=True,
    download=True,
    transform=noise_transform
)
noisy_test_dataset = torchvision.datasets.CIFAR10(
    root="data/noisy",
    train=False,
    download=True,
    transform=noise_transform
)

# DataLoaders for the noisy datasets
noisy_train_load = DataLoader(noisy_train_dataset, batch_size=64, shuffle=True)
noisy_test_load = DataLoader(noisy_test_dataset, batch_size=64)