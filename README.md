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
