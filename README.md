
Repository with general Neural Networks using python
## 🚀# NN_python: Quick Start

```python
from src.engine import RedNeuronal

# Crear una red con 2 entradas, 8 neuronas ocultas y 1 salida
red = RedNeuronal([2, 8, 1], f_activacion='sigmoide')

# Entrenar con tus datos
red.entrenar(X_train, Y_train, epocas=1000, lr=0.05)
