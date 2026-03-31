"""
=============================================================
Role 2 — Intermediate Model (CNN)
Project: CIFAR-10 Image Classification Benchmark
File   : training/train_cnn.py
=============================================================
Responsibilities:
  1. Train CNN model
  2. Compare optimizers (SGD, Adam, RMSprop, Adagrad)
  3. Compare CNN with ANN
  4. Save learning curves and results plots
=============================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report, confusion_matrix

# ── Import CNN model from models/ ───────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.cnn_model import build_cnn, CLASS_NAMES

# ── Reproducibility ─────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── Output directories ───────────────────────────────────────────────
PLOTS_DIR  = os.path.join('results', 'plots')
MODELS_DIR = os.path.join('results', 'models')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 64


# ==================================================================
# 1. Load and preprocess CIFAR-10
# ==================================================================
def load_data():
    """Load CIFAR-10, normalize, and split into train/val/test."""
    (X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

    # Normalize pixel values to [0, 1]
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test       = X_test.astype('float32')       / 255.0

    # 90% train / 10% validation
    val_size = int(0.1 * len(X_train_full))
    X_val    = X_train_full[:val_size]
    y_val    = y_train_full[:val_size]
    X_train  = X_train_full[val_size:]
    y_train  = y_train_full[val_size:]

    print('Data loaded successfully')
    print(f'  Train : {X_train.shape}')
    print(f'  Val   : {X_val.shape}')
    print(f'  Test  : {X_test.shape}')
    return X_train, y_train, X_val, y_val, X_test, y_test


# ==================================================================
# 2. Train one model with a given optimizer
# ==================================================================
def train_model(optimizer, X_train, y_train, X_val, y_val, name=''):
    """Build, compile, and train a CNN with the given optimizer."""
    model = build_cnn()
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=0
    )

    print(f'\n{"="*50}')
    print(f'  Training with optimizer: {name}')
    print(f'{"="*50}')

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    return model, history


# ==================================================================
# 3. Compare optimizers
# ==================================================================
def compare_optimizers(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train CNN with 4 optimizers and collect results."""

    optimizers = {
        'SGD':     tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'Adam':    tf.keras.optimizers.Adam(learning_rate=0.001),
        'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.001),
        'Adagrad': tf.keras.optimizers.Adagrad(learning_rate=0.01),
    }

    all_histories = {}
    all_results   = {}

    for name, opt in optimizers.items():
        model, history = train_model(opt, X_train, y_train, X_val, y_val, name)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        all_histories[name] = history
        all_results[name]   = {
            'model':      model,
            'test_acc':   test_acc,
            'test_loss':  test_loss,
            'epochs_run': len(history.history['loss'])
        }
        print(f'  → Test Accuracy : {test_acc:.4f}')
        print(f'  → Test Loss     : {test_loss:.4f}')
        print(f'  → Epochs run    : {len(history.history["loss"])}/{EPOCHS}')

    # ── Print summary table ──────────────────────────────────────────
    print(f'\n{"="*55}')
    print(f'  {"Optimizer":<12} {"Test Acc":>12} {"Test Loss":>12} {"Epochs":>8}')
    print(f'  {"-"*51}')
    for name, res in all_results.items():
        print(f'  {name:<12} {res["test_acc"]:>12.4f} '
              f'{res["test_loss"]:>12.4f} {res["epochs_run"]:>8}')
    print(f'{"="*55}')

    best_name = max(all_results, key=lambda k: all_results[k]['test_acc'])
    print(f'\n  Best optimizer: {best_name} '
          f'(acc = {all_results[best_name]["test_acc"]:.4f})')

    return all_histories, all_results, best_name


# ==================================================================
# 4. Plots
# ==================================================================
COLORS = {
    'SGD':     '#888780',
    'Adam':    '#7F77DD',
    'RMSprop': '#1D9E75',
    'Adagrad': '#BA7517'
}


def plot_optimizer_comparison(all_histories):
    """Plot val accuracy and val loss for all optimizers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Optimizer Comparison — CNN on CIFAR-10',
                 fontsize=13, fontweight='bold')

    for name, hist in all_histories.items():
        c = COLORS[name]
        axes[0].plot(hist.history['val_accuracy'],
                     label=name, color=c, linewidth=2)
        axes[1].plot(hist.history['val_loss'],
                     label=name, color=c, linewidth=2)

    for ax, title, ylabel in zip(
        axes,
        ['Validation Accuracy per Epoch', 'Validation Loss per Epoch'],
        ['Accuracy', 'Loss']
    ):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'optimizer_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {path}')


def plot_learning_curves(history, optimizer_name):
    """Plot train vs val accuracy and loss for the best model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'CNN Learning Curves — {optimizer_name}',
                 fontsize=13, fontweight='bold')

    c_acc  = '#7F77DD'
    c_loss = '#D85A30'

    # Accuracy
    axes[0].plot(history.history['accuracy'],
                 label='Train', color=c_acc, linewidth=2)
    axes[0].plot(history.history['val_accuracy'],
                 label='Validation', color=c_acc, linewidth=2, linestyle='--')
    axes[0].fill_between(
        range(len(history.history['accuracy'])),
        history.history['accuracy'],
        history.history['val_accuracy'],
        alpha=0.08, color=c_acc
    )
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],
                 label='Train', color=c_loss, linewidth=2)
    axes[1].plot(history.history['val_loss'],
                 label='Validation', color=c_loss, linewidth=2, linestyle='--')
    axes[1].fill_between(
        range(len(history.history['loss'])),
        history.history['loss'],
        history.history['val_loss'],
        alpha=0.08, color=c_loss
    )
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'cnn_learning_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {path}')


def plot_confusion_matrix(model, X_test, y_test, title='CNN'):
    """Plot confusion matrix for the best model."""
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = y_test.flatten()

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label',      fontsize=12)
    ax.set_title(f'Confusion Matrix — {title}', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, 'cnn_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {path}')

    print(f'\nClassification Report — {title}')
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    return y_pred


def plot_cnn_vs_ann(ann_history, cnn_history, ann_acc, cnn_acc, best_name):
    """Bar + line chart comparing ANN vs best CNN."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('CNN vs ANN — CIFAR-10', fontsize=13, fontweight='bold')

    # Line: val accuracy over epochs
    axes[0].plot(ann_history.history['val_accuracy'],
                 label='ANN', color='#378ADD', linewidth=2, linestyle='--')
    axes[0].plot(cnn_history.history['val_accuracy'],
                 label=f'CNN ({best_name})', color='#7F77DD', linewidth=2)
    axes[0].set_title('Validation Accuracy per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Bar: final test accuracy
    bars = axes[1].bar(
        [f'ANN', f'CNN\n({best_name})'],
        [ann_acc, cnn_acc],
        color=['#378ADD', '#7F77DD'],
        width=0.45, edgecolor='none'
    )
    for bar, acc in zip(bars, [ann_acc, cnn_acc]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{acc:.4f}', ha='center', va='bottom', fontweight='bold'
        )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title('Final Test Accuracy')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'cnn_vs_ann_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {path}')


# ==================================================================
# 5. ANN baseline (for comparison only)
# ==================================================================
def build_ann():
    from tensorflow.keras import layers, models
    model = models.Sequential(name='ANN_CIFAR10')
    model.add(layers.Flatten(input_shape=(32, 32, 3)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10,  activation='softmax'))
    return model


def train_ann(X_train, y_train, X_val, y_val, X_test, y_test):
    print('\nTraining ANN baseline for comparison...')
    model = build_ann()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, verbose=0)],
        verbose=1
    )
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'  ANN Test Accuracy: {acc:.4f}')
    return model, history, acc


# ==================================================================
# MAIN
# ==================================================================
if __name__ == '__main__':

    # 1. Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # 2. Compare optimizers
    print('\n--- Optimizer Comparison ---')
    all_histories, all_results, best_name = compare_optimizers(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # 3. Best model & its results
    best_model   = all_results[best_name]['model']
    best_history = all_histories[best_name]
    best_acc     = all_results[best_name]['test_acc']

    # 4. Save best model
    model_path = os.path.join(MODELS_DIR, f'cnn_{best_name.lower()}.h5')
    best_model.save(model_path)
    print(f'\n  Best model saved: {model_path}')

    # 5. Plots
    print('\n--- Generating plots ---')
    plot_optimizer_comparison(all_histories)
    plot_learning_curves(best_history, best_name)
    plot_confusion_matrix(best_model, X_test, y_test,
                          title=f'CNN ({best_name})')

    # 6. CNN vs ANN comparison
    print('\n--- CNN vs ANN ---')
    _, ann_history, ann_acc = train_ann(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    plot_cnn_vs_ann(ann_history, best_history, ann_acc, best_acc, best_name)

    print(f'\n{"="*50}')
    print('  All done!')
    print(f'  ANN  accuracy : {ann_acc:.4f}')
    print(f'  CNN  accuracy : {best_acc:.4f}  (optimizer: {best_name})')
    print(f'  Improvement   : +{(best_acc - ann_acc)*100:.2f}%')
    print(f'{"="*50}')