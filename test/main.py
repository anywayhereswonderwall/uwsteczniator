from uwsteczniatorV2.engine import Tensor
import numpy as np

# Test 1
w = Tensor(np.array([[0.67, 0.41, 0.05, 0.01]]))
x = Tensor(np.array([[0.5], [0.8], [0.3], [0.4]]))
b = Tensor(np.array([[0.1]]))

last = float("inf")
for _ in range(50):
    y = w * x + b
    z = y.sigmoid()
    L = (z - Tensor(np.array([[0.5]])))**2
    L.back()
    w.array -= 0.3 * w.grad
    b.array -= 0.3 * b.grad
    w.grad = np.zeros(w.shape)
    b.grad = np.zeros(b.shape)
    assert L.array[0][0] < last
    last = L.array[0][0]

# Test 2
