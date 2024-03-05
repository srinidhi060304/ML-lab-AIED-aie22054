import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron function
def perceptron(x, w):
    return step_function(np.dot(x, w[1:]) + w[0])

# Error calculation function
def calculate_error(X, y, w):
    error = 0
    for i in range(len(X)):
        prediction = perceptron(X[i], w)
        error += (y[i] - prediction) ** 2
    return error

# Training function
def train_perceptron(X, y, w, learning_rate, max_epochs, convergence_error):
    errors = []
    epoch = 0
    while epoch < max_epochs:
        total_error = calculate_error(X, y, w)
        errors.append(total_error)
        if total_error <= convergence_error:
            break
        for i in range(len(X)):
            prediction = perceptron(X[i], w)
            error = y[i] - prediction
            w[1:] += learning_rate * error * X[i]
            w[0] += learning_rate * error
        epoch += 1
    return w, errors

# Load data
data = pd.read_excel("C:\\Users\\admin\\Desktop\\sem4\\training_mathbert 1.xlsx")

# Extract features
X = data.filter(like='embed_').values

# Add bias term to input data
X_bias = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones at the beginning of X

# Extract target classes
y = data['output'].values

# Initial weights
num_features = X_bias.shape[1]  # Corrected: use X_bias.shape instead of X.shape
W = np.zeros(num_features)  # Initialize weights to zeros, including the bias term

# Parameters
learning_rate = 0.05
max_epochs = 1000
convergence_error = 0.002

# Train the perceptron
trained_weights, errors = train_perceptron(X_bias, y, W, learning_rate, max_epochs, convergence_error)

# Plot epochs against error values
plt.plot(range(1, len(errors) + 1), errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs')
plt.show()

# Print trained weights
print("Trained Weights:", trained_weights)
