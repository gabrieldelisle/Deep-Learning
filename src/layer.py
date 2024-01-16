from abc import ABC

import numpy as np

EPSILON = 1e-12


class Layer(ABC):
    def forward(self, X):
        raise NotImplementedError

    def backward(self, g):
        raise NotImplementedError

    def step(self, eta):
        raise NotImplementedError

    def predict(self, X):
        return self.forward(X)

    def cost(self):
        return 0


class DenseLayer(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        regularization_factor: int = 1,
    ):
        # Xavier initialisation
        sigma = 2 / np.sqrt(input_size)
        self.W = np.random.normal(0, sigma, (output_size, input_size))
        self.b = np.random.normal(0, sigma, (output_size, 1))
        self.regularization_factor = regularization_factor

    def forward(self, X):
        self.X = X
        return self.W.dot(X) + self.b

    def backward(self, g):
        self.grad_W = g.dot(self.X.T) + 2 * self.regularization_factor * self.W
        self.grad_b = g.dot(np.ones((self.X.shape[1], 1)))

        return self.W.T.dot(g)

    def step(self, eta):
        self.W -= eta * self.grad_W
        self.b -= eta * self.grad_b

    def cost(self):
        return self.regularization_factor * np.sum(self.W**2)


class ReLuLayer(Layer):
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, g):
        return g * (self.X > 0)

    def step(self, eta):
        pass


class BatchNormalizationLayer(Layer):
    def __init__(self, size, regularization_factor=1, alpha=0.9):
        # Xavier initialisation
        sigma = 2 / np.sqrt(size)
        self.gamma = np.random.normal(0, sigma, (size, 1))
        self.beta = np.random.normal(0, sigma, (size, 1))
        self.regularization_factor = regularization_factor
        self.alpha = alpha
        self.mu_av = None
        self.v_av = None

    def forward(self, X):
        self.X = X

        self.mu = np.mean(X, axis=1).reshape(-1, 1)
        self.v = np.mean(X**2, axis=1)

        if self.mu_av is None:
            self.mu_av = self.mu
        else:
            self.mu_av = self.alpha * self.mu_av + (1 - self.alpha) * self.mu

        if self.v_av is None:
            self.v_av = self.v
        else:
            self.v_av = self.alpha * self.v_av + (1 - self.alpha) * self.v

        X = (np.diag((self.v + EPSILON) ** -0.5)).dot(X - self.mu)
        return self.gamma * X + self.beta

    def backward(self, g):
        N = g.shape[1]
        oneN = np.ones((N, 1))

        self.grad_gamma = (g * self.X).dot(oneN)
        self.grad_beta = g.dot(oneN)

        g *= self.gamma

        sigma1 = (self.v + EPSILON) ** -0.5
        sigma2 = (self.v + EPSILON) ** -1.5
        g1 = g * sigma1.reshape(-1, 1)
        g2 = g * sigma2.reshape(-1, 1)
        D = self.X - self.mu

        return g1 - D * (g2 * D).dot(oneN) / N - g1.dot(oneN) / N

    def step(self, eta):
        self.gamma -= eta * self.grad_gamma
        self.beta -= eta * self.grad_beta

    def predict(self, X):
        X = (np.diag((self.v_av + EPSILON) ** -0.5)).dot(X - self.mu_av)
        return self.gamma * X + self.beta

    def cost(self):
        return self.regularization_factor * np.sum(self.W**2)
