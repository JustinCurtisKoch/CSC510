# CSC510 Foundations of Artificial Intelligence
# Module 3: Hand-Made Shallow ANN in Python

import numpy as np

# Define layers by number of neurons
layer_input = 3 # Input layer with 3 neurons
layer_hidden = 4 # Hidden layer with 4 neurons
layer_output = 1 # Output layer with 1 neurons


# Initialize weights and biases
# Input data 
x = np.array([[3], [5], [7]]) # shape (3, 1)
# w1: connects input → hidden
w1 = np.random.randn(layer_hidden, layer_input)   # shape (4, 3)

# b1: bias for hidden layer
b1 = np.random.randn(layer_hidden, 1)      # shape (4, 1)

# w2: connects hidden → output
w2 = np.random.randn(layer_output, layer_hidden)  # shape (1, 4)

# b2: bias for output layer
b2 = np.random.randn(layer_output, 1)         # shape (1, 1)

# Placeholders for forward pass variables
z1 = None
a1 = None
z2 = None
a2 = None

# Forward pass function
def forward_pass(x):
    global z1, a1, z2, a2
    
    # Compute hidden layer activation
    z1 = np.dot(w1, x) + b1  # shape (4, 1)
    a1 = np.maximum(0, z1)   # ReLU activation function

    # Compute output layer activation
    z2 = np.dot(w2, a1) + b2  # shape (1, 1)
    a2 = z2  # No activation function for output layer in this example

# Loss function (Mean Squared Error)
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Example run
x = np.array([[3], [5], [7]])  # Input
y = np.array([[9]])            # Expected output

forward_pass(x)
loss = compute_loss(y, a2)

print("Prediction:", a2)
print("Loss:", loss)
