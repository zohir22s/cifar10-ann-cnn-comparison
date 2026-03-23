#Optionally, you can run this script to download the raw CIFAR-10 dataset using torchvision.

import torchvision

# download raw train dataset
torchvision.datasets.CIFAR10(
    root="data/raw",
    train=True,
    download=True
)

# download raw test dataset
torchvision.datasets.CIFAR10(
    root="data/raw",
    train=False,
    download=True
)

