import math
import numpy as np

class Value:

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0  # 0 means no effect meaning we assume that at init it doesnt affect the output
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        output = Value(self.data + other.data, (self, other), "+")
        return output

    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other), "*")
        return output

    def tanh(self):
        n = self.data
        t = (math.exp(2*n)-1)/(math.exp(2*n)+1)
        out = Value(t, _children=(self, ), _op="tanh")
        return out