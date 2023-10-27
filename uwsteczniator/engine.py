import math


# Wrapper for a numeric value, it keeps track of its "history"
class Value:
    def __init__(self, value, children=tuple(), operation=""):
        # numeric value
        self.value = value
        # children (of class Value)
        self.prev = set(children)
        # mathematical expression which resulted in THIS value
        # e.g. a = Value(2) + Value(3) -> a.operation = "+"
        self.operation = operation
        # accumulator for gradients, this will be set from parent node
        self.grad = 0
        #
        self._back = lambda: None

    def back(self):
        # do the topological sort to go from the last value obtained in the sequence of calculations (go backwards)
        self.grad = 1
        nodes = []
        visited = set()

        def topo(node):
            assert isinstance(node, Value), "node must be of type Value"

            if node not in visited:
                visited.add(node)
                for child in node.prev:
                    topo(child)
                nodes.append(node)

        topo(self)
        nodes = reversed(nodes)

        for node in nodes:
            node._back()

    def tanh(self):
        this_value = self.value
        t = ((math.exp(this_value) - math.exp(-this_value)) / (math.exp(this_value) + math.exp(-this_value)))
        out = Value(t, (self,), "tanh")

        def _back():
            self.grad += (1 - t**2) * out.grad

        out._back = _back

        return out

    def exp(self):
        this_value = self.value
        exp = math.exp(this_value)
        out = Value(exp, (self,), "exp")

        def _back():
            self.grad += exp * out.grad
        out._back = _back

        return out

    def __pow__(self, power):
        this_value = self.value
        p = this_value**power
        out = Value(p, (self, ), f"**{power}")

        def _back():
            self.grad = power * (this_value**(power - 1)) * out.grad
        out._back = _back

        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value + other.value, children=(self, other), operation="+")

        def _back():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._back = _back

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value * other.value, children=(self, other), operation="*")

        def _back():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad

        out._back = _back

        return out

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    # without rmul if someone calls (2 * a) it will result in an error
    # because mul checks only the "other" object and in that case it is "value" object
    # ,so it will proceed to :
    # out = Value(self.value * other.value, children=(self, other), operation="*")
    # but here it tries to call .value on an integer, therefore it fails
    # rmul changes the order of multiplication and calls mul:
    # 2 * a -> a * 2
    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value: {self.value}"