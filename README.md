# NeuroVision-MNIST

🚀 Project Overview

This project implements a Multi-Layer Perceptron (MLP) using PyTorch to classify images from the MNIST dataset. The dataset consists of handwritten digits (0-9), and each image is preprocessed into a 1D vector before being fed into the neural network.

To enhance generalization and reduce overfitting, the model incorporates:

Data Augmentation (rotation, scaling, etc.)

Dropout & Batch Normalization layers

Custom Weight Initialization

Hyperparameter Tuning

Additionally, the model's performance is evaluated using:

Accuracy & Loss Trends

Precision, Recall, and F1-Score

Confusion Matrix Analysis

📂 Dataset Preprocessing

Load MNIST Dataset: Download and prepare the dataset using PyTorch's torchvision.datasets.

Flatten Images: Convert each 28x28 grayscale image into a 1D vector (size 784).

Apply Transformations:

Rotation (random angle)

Scaling (zoom in/out)

Normalization (zero mean, unit variance)

Create Data Loaders: Efficient batching and shuffling using PyTorch's DataLoader.

🔧 Model Architecture

The MLP consists of multiple fully connected layers with the following components:

Input Layer: 784 neurons (corresponding to the flattened image pixels)

Hidden Layers:

Fully connected layers with ReLU activation

Batch Normalization to stabilize training

Dropout for regularization

Output Layer: 10 neurons with Softmax activation (one for each digit)

🎯 Custom Weight Initialization

A specialized function is used to initialize model weights, ensuring stable convergence and improved performance.

📊 Model Evaluation

🔥 Performance Metrics

Accuracy: Measures correct predictions

Loss Analysis: Tracks model learning

Precision & Recall: Evaluates class-wise performance

F1-Score: Balances precision & recall

Confusion Matrix: Visualizes misclassifications

📌 Overfitting Analysis

Comparison of training vs. validation loss trends

Effect of dropout & batch normalization on model performance

⚡ Hyperparameter Tuning

To optimize performance, the following hyperparameters are tuned using grid/random search:

Learning Rate

Batch Size

Number of Hidden Layers & Neurons

Dropout Rate

Weight Initialization Strategies
