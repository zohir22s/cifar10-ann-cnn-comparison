# load_data.py
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
# Load the CIFAR-10 dataset from data/raw directory
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
# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)