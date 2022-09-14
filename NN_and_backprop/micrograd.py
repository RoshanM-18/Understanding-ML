import math
import numpy as np

class Value:

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0  # 0 means no effect meaning we assume that at init it doesnt affect the output
        self._prev = set(_children)
        self._backward = lambda: None # does not do anything, empty function
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        output = Value(self.data + other.data, (self, other), "+")

        def _backward():  
            self.grad = 1.0 * output.grad
            other.grad = 1.0 * output.grad

        output._backward = _backward

        return output

    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = other.data * output.grad
            other.grad = self.data * output.grad

        output._backward = _backward

        return output

    def tanh(self):
        n = self.data
        t = (math.exp(2*n)-1)/(math.exp(2*n)+1)
        out = Value(t, _children=(self, ), _op="tanh")

        def _backward():
            self.grad = (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
            
                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()