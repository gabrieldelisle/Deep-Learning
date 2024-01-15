from abc import ABC

import numpy as np


class Layer(ABC):
    def forward(self, X):
        raise NotImplementedError

    def backward(self, g):
        raise NotImplementedError

    def step(self, eta):
        raise NotImplementedError


class DenseLayer(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        normalization_factor: int = 1,
    ):
        # Xavier initialisation
        sigma = 2 / np.sqrt(input_size)
        self.W = np.random.normal(0, sigma, (output_size, input_size))
        self.b = np.random.normal(0, sigma, (output_size, 1))
        self.normalization_factor = normalization_factor

    def forward(self, X):
        self.X = X
        return self.W.dot(X) + self.b

    def backward(self, g):
        self.grad_W = g.dot(self.X.T) + 2 * self.normalization_factor * self.W
        self.grad_b = g.dot(np.ones((self.X.shape[1], 1)))

        return self.W.T.dot(g)

    def step(self, eta):
        self.W -= eta * self.grad_W
        self.b -= eta * self.grad_b


class SoftMaxLayer:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, g):
        return g * (self.X > 0)

    def step(self, eta):
        pass
