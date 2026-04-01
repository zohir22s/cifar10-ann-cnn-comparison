# cifar10-ann-cnn-comparison

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.14-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zohir22s/cifar10-ann-cnn-comparison)

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

### ANN (Artificial Neural Network) — PyTorch
A simple feedforward neural network for CIFAR-10 classification:

- **Input:** 32 × 32 × 3 images, flattened to a 3072-element vector  
- **Architecture:**  
  - Dense(3072 → 1024) + ReLU  
  - Dense(1024 → 512) + ReLU  
  - Dense(512 → 256) + ReLU  
  - Dense(256 → 10) (output logits)  
- **Features:** Fully connected, no convolution, trained on raw pixels.  

---

### CNN (Convolutional Neural Network) — TensorFlow/Keras
An intermediate convolutional model to capture spatial features:

- **Input:** 32 × 32 × 3 images  
- **Architecture:**  
  - **Block 1:** Conv2D(32) → Conv2D(32) → MaxPool(2×2) → Dropout(0.25)  
  - **Block 2:** Conv2D(64) → Conv2D(64) → MaxPool(2×2) → Dropout(0.25)  
  - **Classifier Head:** Flatten → Dense(512, ReLU) → Dropout(0.5) → Dense(10, Softmax)  
- **Features:** Convolutional layers extract spatial features; dropout prevents overfitting.  

---

### Advanced CNN — PyTorch
A deeper CNN with residual connections and regularization:

- **Input:** 32 × 32 × 3 images  
- **Architecture:**  
  - **Stem:** Conv2D(3 → 64) + BatchNorm + ReLU  
  - **Residual Blocks:**  
    - Block1: 64 → 128, stride=2  
    - Block2: 128 → 256, stride=2  
    - Block3: 256 → 512, stride=2  
  - Global Average Pooling → Dropout(0.4) → Dense(10)  
- **Features:**  
  - Residual skip connections improve gradient flow  
  - Batch normalization and dropout for better generalization  
  - Designed for higher accuracy on CIFAR-10  

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

Install all required Python packages using:

```bash
pip install -r requirements.txt

---

## License

This project is licensed under the [MIT License](LICENSE).  
See the `LICENSE` file for more details.