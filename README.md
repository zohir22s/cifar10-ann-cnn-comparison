# cifar10-ann-cnn-comparison

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Notebook](https://img.shields.io/badge/notebook-ready-orange.svg)](https://colab.research.google.com/github/zohir22s/cifar10-ann-cnn-comparison)

This repository contains implementations of **ANN**, **CNN**, and **Advanced CNN** for image classification on the **CIFAR-10 dataset**. It provides training scripts, evaluation tools, metrics, and notebooks for analysis and comparison.

---

## Dataset

The project uses the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

---

## Run Notebooks

### Exploratory Data Analysis (EDA)
[![EDA Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zohir22s/cifar10-ann-cnn-comparison/blob/main/notebooks/eda.ipynb)

### Experiments
[![Experiments Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zohir22s/cifar10-ann-cnn-comparison/blob/main/notebooks/experiments.ipynb)

---

## Models

- **ANN** – Simple fully connected network (PyTorch)  
- **CNN** – Intermediate convolutional network (TensorFlow/Keras)  
- **Advanced CNN** – PyTorch CNN with data augmentation, dropout, and learning rate scheduling  

---

## Features

- Train and evaluate **ANN**, **CNN**, and **Advanced CNN**  
- Save best and last model checkpoints  
- Evaluate with **accuracy, precision, recall, F1-score**  
- Generate and save **confusion matrices**  
- Compare CNN optimizers: SGD, Adam, RMSprop, Adagrad  
- Jupyter notebooks for **model comparison and analysis**  

---

## Requirements

```text
# Core PyTorch and TensorFlow libraries
torch
torchvision
tensorflow
keras

# Data handling and analysis
numpy
pandas
scikit-learn

# Visualization
matplotlib
seaborn
plotly

# Progress bars
tqdm

# Jupyter notebooks
notebook

# Optional: saving/loading Python objects
pickle5