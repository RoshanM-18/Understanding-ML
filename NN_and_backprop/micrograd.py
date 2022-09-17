import math

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

        other = other if isinstance(other, Value) else Value(other) # to do operations like Value(a) + 1
        output = Value(self.data + other.data, (self, other), "+")

        def _backward():  
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad

        output._backward = _backward

        return output

    def __radd__(self, other): # other + self

        return self + other

    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other) # to do operations like Value(a) * 3
        output = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward

        return output

    def __pow__(self, other):

        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f"**(other)")

        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other): # other * self

        return self * other

    def __truediv__(self, other): 

        return self * other**-1

    def __neg__(self): # -self

        return self * -1

    def __sub__(self, other): # self - other

        return self + (-other)

    def exp(self):

        x = self.data
        out = Value(math.exp(x), (self, ), "exp")

        def _backward():
            self.grad += out.data * out.grad

        self._backward = _backward

        return out

    def tanh(self):
        
        n = self.data
        t = (math.exp(2*n)-1)/(math.exp(2*n)+1)
        out = Value(t, _children=(self, ), _op="tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

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