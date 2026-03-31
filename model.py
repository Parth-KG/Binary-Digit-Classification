import numpy as np

class BinaryNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights (small random numbers) and biases (zeros)
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        # Returns 1 if Z > 0, else 0
        return Z > 0

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        # Forward Prop
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)
        
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def compute_loss(self, A2, Y):
        # Binary Cross Entropy Loss
        m = Y.shape[1]
        # Add a tiny epsilon (1e-8) to avoid log(0) error
        logprobs = np.multiply(np.log(A2 + 1e-8), Y) + np.multiply(np.log(1 - A2 + 1e-8), 1 - Y)
        loss = -np.sum(logprobs) / m
        return loss

    def backward(self, X, Y, cache):
        m = X.shape[1] # Number of examples
        # Retrieve internals from forward pass
        A1 = cache["A1"]
        A2 = cache["A2"]
        Z1 = cache["Z1"]
        
        # 1. Output Layer Gradients
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # 2. Hidden Layer Gradients
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.relu_derivative(Z1) # Element-wise multiply
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads