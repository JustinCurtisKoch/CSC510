# CSC510 Foundations of Artificial Intelligence
# Module 3: Hand-Made Shallow ANN in Python

import numpy as np

# Define layers by number of neurons
layer_input = 3 # Input layer with 3 neurons
layer_hidden = 4 # Hidden layer with 4 neurons
layer_output = 1 # Output layer with 1 neurons

# Input data 
x = np.array([[3], [5], [7]]) # shape (3, 1)
y_true = np.array([[9]])   # (1, 1) target

# Initialize weights and biases
np.random.seed(42)
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
    return a2
# Forward pass to get predictions
forward_pass(x)

# Loss function (Mean Squared Error)
def compute_loss(y_true, a2):
    return np.mean((y_true - a2) ** 2)

loss = compute_loss(y_true, a2)
print("Prediction:", a2)
print("Loss:", loss)

# Backward propagation function
def backward_pass(y_true):
    global w1, b1, w2, b2

    # ===== Output layer gradients =====
    # Derivative of MSE loss wrt a2
    d_loss_a2 = 2 * (a2 - y_true) / y_true.size  # shape (1, m)
    
    # Derivative of a2 wrt z2 (Identity activation → derivative = 1)
    d_a2_z2 = 1  
    
    # Gradients wrt w2 and b2
    grad_w2 = np.dot(d_loss_a2 * d_a2_z2, a1.T)  # shape (1, hidden_units)
    grad_b2 = np.sum(d_loss_a2 * d_a2_z2, axis=1, keepdims=True)  # shape (1, 1)

    # ===== Hidden layer gradients =====
    # Derivative of ReLU wrt z1
    d_a1_z1 = np.where(z1 > 0, 1, 0)  
    
    # Backpropagate from output to hidden layer
    d_loss_a1 = np.dot(w2.T, d_loss_a2 * d_a2_z2) * d_a1_z1  # shape (hidden_units, m)
    
    # Gradients wrt w1 and b1
    grad_w1 = np.dot(d_loss_a1, x.T)  # shape (hidden_units, input_dim)
    grad_b1 = np.sum(d_loss_a1, axis=1, keepdims=True)  # shape (hidden_units, 1)

    # ===== Parameter update =====
    learning_rate = 0.01
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * grad_b1
    w2 -= learning_rate * grad_w2
    b2 -= learning_rate * grad_b2

    # === TEST RUN ===
print("Before training:")
print("Prediction:", forward_pass(x))
print("Loss:", np.mean((a2 - y_true) ** 2))

backward_pass(y_true)

print("\nAfter backpropagation update:")
print("Updated Prediction:", forward_pass(x))
print("Updated Loss:", np.mean((a2 - y_true) ** 2))

epochs = 1000
for i in range(epochs):
    forward_pass(x)
    loss = compute_loss(y_true, a2)
    backward_pass(y_true)
    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss}, Prediction: {a2}")

# Final output
final_pred = forward_pass(x)
print("\nFinal prediction:", final_pred.flatten())
print("Final loss:", compute_loss(y_true, final_pred))
