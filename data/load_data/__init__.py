# data/load_data.py
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# dataset directory
DATA_DIR = "data/raw"

# create folder if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# check if dataset already exists
dataset_exists = os.path.exists(os.path.join(DATA_DIR, "cifar-10-batches-py"))

# Load datasets (download only if missing)
train_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=not dataset_exists,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=not dataset_exists,
    transform=transform
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)