import numpy as np

class Layer:
    """
    Represents a single layer in the neural network.
    """
    def __init__(self, n_inputs, n_neurons, activation_function):
        # Xavier/Glorot initialization for better convergence
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / (n_inputs + n_neurons))
        self.bias = np.zeros((1, n_neurons))
        self.activation_function = activation_function
        self.last_input = None
        self.last_output = None
        self.gradient_w = None
        self.gradient_b = None

class NeuralNetwork:
    """
    A Feedforward Neural Network built from scratch using NumPy.
    """
    def __init__(self, layers_dim, activation='sigmoid'):
        self.layers = []
        self.activation_name = activation
        
        # Build the architecture
        for i in range(len(layers_dim) - 1):
            self.layers.append(Layer(layers_dim[i], layers_dim[i+1], activation))

    def _activate(self, x, derivative=False):
        if self.activation_name == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s) if derivative else s
        
        if self.activation_name == 'tanh':
            t = np.tanh(x)
            return (1 - t**2) if derivative else t
        
        return x

    def predict(self, x):
        """Forward propagation"""
        output = x
        for layer in self.layers:
            layer.last_input = output
            # Z = W * X + b
            z = np.dot(output, layer.weights) + layer.bias
            layer.last_output = self._activate(z)
            output = layer.last_output
        return output

    def train(self, x_train, y_train, epochs=1000, lr=0.01):
        """Training loop using backpropagation"""
        for epoch in range(epochs):
            # 1. Forward pass
            predictions = self.predict(x_train)
            
            # 2. Compute error (MSE derivative)
            error = predictions - y_train
            
            # 3. Backward pass
            delta = error
            for layer in reversed(self.layers):
                # Gradient calculation
                d_activation = self._activate(layer.last_output, derivative=True)
                delta_layer = delta * d_activation
                
                layer.gradient_w = np.dot(layer.last_input.T, delta_layer)
                layer.gradient_b = np.sum(delta_layer, axis=0, keepdims=True)
                
                # Update delta for the next layer (moving backwards)
                delta = np.dot(delta_layer, layer.weights.T)
                
                # 4. Update weights and biases (Gradient Descent)
                layer.weights -= lr * layer.gradient_w
                layer.bias -= lr * layer.gradient_b

            # Optional: Print loss every 10% of epochs
            if epoch % (epochs // 10) == 0:
                loss = np.mean((predictions - y_train) ** 2)
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.6f}")

    def save_model(self, filename):
        """Saves weights and biases to a file"""
        data = {f"layer_{i}": {"w": l.weights, "b": l.bias} for i, l in enumerate(self.layers)}
        np.save(filename, data)
        print(f"Model saved to {filename}")
