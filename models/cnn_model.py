"""
=============================================================
Role 2 — Intermediate Model (CNN)
Project: CIFAR-10 Image Classification Benchmark
File   : models/cnn_model.py
=============================================================
CNN Architecture for CIFAR-10
------------------------------
Input  : 32 x 32 x 3  (RGB image)
Output : 10 classes   (softmax probabilities)

Architecture:
  Block 1 : Conv2D(32) → Conv2D(32) → MaxPool(2x2) → Dropout(0.25)
  Block 2 : Conv2D(64) → Conv2D(64) → MaxPool(2x2) → Dropout(0.25)
  Head    : Flatten → Dense(512, ReLU) → Dropout(0.5) → Dense(10, Softmax)
=============================================================
"""

import tensorflow as tf
from tensorflow.keras import layers, models


# ------------------------------------------------------------------
# Class Names (CIFAR-10)
# ------------------------------------------------------------------
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

NUM_CLASSES  = 10
INPUT_SHAPE  = (32, 32, 3)


# ------------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------------
def build_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Build and return the CNN model for CIFAR-10 classification.

    Parameters
    ----------
    input_shape : tuple, default (32, 32, 3)
        Shape of a single input image (H, W, C).
    num_classes : int, default 10
        Number of output classes.

    Returns
    -------
    model : tf.keras.Sequential
        Compiled-ready CNN model (not yet compiled).
    """

    model = models.Sequential(name='CNN_CIFAR10')

    # ── Block 1 ─────────────────────────────────────────────────────
    # Two consecutive Conv layers learn low-level features (edges, colours).
    # padding='same' keeps spatial dimensions at 32x32 after convolution.
    model.add(layers.Conv2D(
        filters=32, kernel_size=(3, 3),
        activation='relu', padding='same',
        input_shape=input_shape,
        name='conv1_1'
    ))
    model.add(layers.Conv2D(
        filters=32, kernel_size=(3, 3),
        activation='relu', padding='same',
        name='conv1_2'
    ))
    # MaxPool halves spatial dimensions: 32x32 → 16x16
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool1'))
    # Dropout prevents overfitting by randomly dropping 25% of neurons
    model.add(layers.Dropout(rate=0.25, name='drop1'))

    # ── Block 2 ─────────────────────────────────────────────────────
    # Deeper filters (64) capture higher-level features (shapes, textures).
    model.add(layers.Conv2D(
        filters=64, kernel_size=(3, 3),
        activation='relu', padding='same',
        name='conv2_1'
    ))
    model.add(layers.Conv2D(
        filters=64, kernel_size=(3, 3),
        activation='relu', padding='same',
        name='conv2_2'
    ))
    # MaxPool: 16x16 → 8x8
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool2'))
    model.add(layers.Dropout(rate=0.25, name='drop2'))

    # ── Classifier Head ─────────────────────────────────────────────
    # Flatten: 8x8x64 = 4096 features → 1D vector
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(units=512, activation='relu', name='dense1'))
    # Higher dropout (50%) in the dense layer to strongly regularize
    model.add(layers.Dropout(rate=0.5, name='drop3'))
    # Softmax outputs a probability distribution over 10 classes
    model.add(layers.Dense(units=num_classes, activation='softmax', name='output'))

    return model


# ------------------------------------------------------------------
# Quick test — run this file directly to verify the model builds
# ------------------------------------------------------------------
if __name__ == '__main__':
    model = build_cnn()
    model.summary()

    print('\nLayer-by-layer output shapes:')
    print(f'  {"Layer":<20} {"Output Shape":<25} {"Params":>10}')
    print('  ' + '-' * 57)
    for layer in model.layers:
        out = str(layer.output_shape)
        params = layer.count_params()
        print(f'  {layer.name:<20} {out:<25} {params:>10,}')

    total = model.count_params()
    print(f'\n  Total parameters: {total:,}')
