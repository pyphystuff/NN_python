import numpy as np
import sys
import os

# This line allows the script to find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import NeuralNetwork

def run_xor_example():
    # 1. Prepare XOR data
    # Inputs: [0,0], [0,1], [1,0], [1,1]
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Targets: 0, 1, 1, 0
    y = np.array([[0], [1], [1], [0]])

    print("--- Training Neural Network for XOR Gate ---")

    # 2. Initialize the network
    # Architecture: 2 inputs -> 4 hidden neurons -> 1 output
    nn = NeuralNetwork(layers_dim=[2, 4, 1], activation='sigmoid')

    # 3. Train the model
    # We use a higher learning rate for this small problem
    nn.train(X, y, epochs=10000, lr=0.1)

    print("\n--- Final Results ---")
    predictions = nn.predict(X)
    
    for i in range(len(X)):
        print(f"Input: {X[i]} | Target: {y[i][0]} | Predicted: {predictions[i][0]:.4f}")

if __name__ == "__main__":
    run_xor_example()
