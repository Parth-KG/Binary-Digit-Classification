import numpy as np
from sklearn.datasets import fetch_openml

def load_binary_mnist():
    print("Downloading MNIST... this may take a while.")
    # Fetch data from OpenML
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X_all = mnist.data  # The images (pixels)
    y_all = mnist.target # The labels (0-9)

    # Convert labels to integers
    y_all = y_all.astype(np.uint8)

    # Filter (選ぶ) only 0 and 1
    # We create a boolean mask where label is 0 OR 1
    mask = (y_all == 0) | (y_all == 1)
    
    X_binary = X_all[mask]
    y_binary = y_all[mask]

    print(f"Dataset shape: {X_binary.shape}") # Should be approx 14k images
    return X_binary, y_binary

if __name__ == "__main__":
    X, y = load_binary_mnist()
    print("Success. Data is ready.")