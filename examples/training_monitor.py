import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure the script can locate the 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import NeuralNetwork

def run_monitor_example():
    print("--- Neural Network Training Monitor ---")

    # 1. Create a synthetic dataset (3D paraboloid)
    # This represents a more complex multi-input regression task
    X = np.random.uniform(-2, 2, (500, 2))
    y = (X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)

    # 2. Initialize the network
    # 2 inputs -> 10 hidden -> 1 output
    nn = NeuralNetwork(layers_dim=[2, 10, 1], activation='tanh')

    # 3. Custom training loop to capture loss history
    epochs = 1000
    learning_rate = 0.01
    loss_history = []

    print("Training in progress...")
    for epoch in range(epochs):
        # We manually call predict and training logic 
        # (or you can modify your engine.py to return loss)
        predictions = nn.predict(X)
        loss = np.mean((predictions - y) ** 2)
        loss_history.append(loss)
        
        # Internal update
        nn.train(X, y, epochs=1, lr=learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | MSE Loss: {loss:.6f}")

    # 4. Plot the Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='#2c3e50', lw=2)
    plt.title('Training Loss Evolution (Learning Curve)', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.yscale('log') # Log scale is better to see convergence
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    print("\nLoss curve generated. This helps visualize convergence speed.")
    plt.show()

if __name__ == "__main__":
    run_monitor_example()
