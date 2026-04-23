import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# This line allows the script to find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import NeuralNetwork

def run_sine_example():
    print("--- Training Neural Network for Sine Wave Regression ---")

    # 1. Generate noisy data
    # 200 points from -pi to pi
    X = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
    # Target is sin(x) with some Gaussian noise
    y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

    # 2. Initialize the network
    # Architecture: 1 input -> 15 hidden -> 15 hidden -> 1 output
    # We use 'tanh' as it's better for regression than sigmoid
    nn = NeuralNetwork(layers_dim=[1, 15, 15, 1], activation='tanh')

    # 3. Train the model
    # More neurons and iterations to capture the curve's complexity
    nn.train(X, y, epochs=3000, lr=0.01)

    # 4. Predict for visualization
    predictions = nn.predict(X)

    # 5. Plotting results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.4, label='Noisy Training Data')
    plt.plot(X, predictions, color='red', lw=3, label='Neural Network Fit')
    plt.title('Non-linear Regression: Function Approximation', fontsize=14)
    plt.xlabel('Input (x)')
    plt.ylabel('Output (sin x)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the result to assets folder (make sure the folder exists)
    # plt.savefig('../assets/sine_plot.png') 
    
    print("\nPlot generated. If running locally, it will display now.")
    plt.show()

if __name__ == "__main__":
    run_sine_example()
