# Neural Networks from Scratch in Python

A lightweight and educational implementation of Feedforward Neural Networks using only **NumPy**. This project demonstrates the fundamental mechanics of backpropagation and gradient descent from a first-principles perspective.

## 🚀 Key Features
* **Modular Architecture**: Layer-based design that allows for flexible network depth and custom neuron counts.
* **Pure NumPy Implementation**: No heavy frameworks like TensorFlow or PyTorch; only linear algebra and calculus.
* **Activation Functions**: Support for Sigmoid and Tanh, including their derivatives for backpropagation.
* **Optimization**: Standard Gradient Descent optimizer with configurable learning rates.

## 🧪 Mathematical Foundation
The network minimizes the Mean Squared Error (MSE) to find the optimal weights and biases:
$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

During the **Backward Pass**, gradients are calculated using the chain rule to update each layer:
$$\Delta w = \eta \cdot (Input^T \cdot \delta)$$

## 📦 Installation & Setup
To run this project locally, clone the repository and install the dependencies:

```bash
git clone [https://github.com/pyphystuff/neural_networks_python.git](https://github.com/pyphystuff/neural_networks_python.git)
cd neural_networks_python
pip install -r requirements.txt

## 📂 Project Structure

```text
neural_networks_python/
├── assets/              # Visual results and plots for documentation
├── examples/            # Ready-to-run demonstration scripts
│   ├── __init__.py
│   ├── xor_gate.py      # Classic XOR logic problem
│   ├── sine_regression.py # Non-linear function approximation
│   └── training_monitor.py # Loss curve and convergence analysis
├── notebooks/           # Interactive Jupyter tutorials
│   └── regression.ipynb
├── src/                 # Core engine (the framework)
│   ├── __init__.py
│   ├── engine.py        # NeuralNetwork and Layer classes
│   └── activations.py   # Activation functions and derivatives
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
