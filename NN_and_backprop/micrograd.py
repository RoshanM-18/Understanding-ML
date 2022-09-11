import numpy as np

class Value:

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        output = self.data + other.data
        return output

    def __mul__(self, other):
        output = self.data * other.data
        return output