import numpy as np


# Wrapper for ndarray
class Tensor:
    def __init__(self, array, children=tuple()):
        # the wrapped ndarray
        self.array = array
        self.prev = set(children)
        self.shape = array.shape
        self.grad = np.zeros(self.shape)

        self._back = lambda: None

    def __add__(self, other):
        assert isinstance(other, Tensor), "You can add tensors only"

        out = Tensor(self.array + other.array, children=(self, other))

        def _back():
            self.grad += out.grad
            other.grad += out.grad
        out._back = _back

        return out

    def __mul__(self, other):
        assert isinstance(other, Tensor), "You can multiply tensors only"

        out = Tensor(np.matmul(self.array, other.array), children=(self, other))

        def _back():
            self.grad += np.matmul(out.grad, other.array.T)
            other.grad += np.matmul(self.array.T, out.grad)
        out._back = _back

        return out

    def tanh(self):
        this_value = self.array
        t = ((np.exp(this_value) - np.exp(-this_value)) / (np.exp(this_value) + np.exp(-this_value)))
        out = Tensor(t, (self,))

        def _back():
            self.grad += (1 - t**2) * out.grad

        out._back = _back

        return out

    def sigmoid(self):
        this_value = self.array
        s = 1 / (1 + np.exp(-this_value))
        out = Tensor(s, (self,))

        def _back():
            self.grad += (s * (1 - s)) * out.grad

        out._back = _back

        return out

    def back(self):
        self.grad = np.ones(self.shape)
        nodes = []
        visited = set()

        def topo(node):
            assert isinstance(node, Tensor), "node must be of type Tensor"

            if node not in visited:
                visited.add(node)
                for child in node.prev:
                    topo(child)
                nodes.append(node)

        topo(self)
        nodes = reversed(nodes)

        for node in nodes:
            node._back()

    def __pow__(self, power):
        this_value = self.array
        p = this_value**power
        out = Tensor(p, (self, ))

        def _back():
            self.grad = power * (this_value**(power - 1)) * out.grad

        out._back = _back

        return out


    def __neg__(self): # -self
        out = Tensor(-self.array, children=(self,))

        def _back():
            self.grad += -1 * out.grad

        out._back = _back

        return out

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __repr__(self):
        return f"Tensor({self.array})"

