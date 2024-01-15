import numpy as np

from .layer import Layer


def softmax(X):
    p = np.exp(X)
    return p / np.sum(p, axis=0)


def generate_batches(X, Y, n_batch):
    N = X.shape[1]

    # shuffle
    indexes = np.arange(N)
    np.random.shuffle(indexes)
    X = X[:, indexes]
    Y = Y[:, indexes]

    for i in range(N // n_batch):
        start = i * n_batch
        end = (i + 1) * n_batch
        yield X[:, start:end], Y[:, start:end]


class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def _step(self, X, Y, eta):
        N = X.shape[1]
        P = self.predict(X)

        # gradiant descent
        g = (P - Y) / N
        for layer in reversed(self.layers):
            g = layer.backward(g)
            layer.step(eta)

    def fit(self, X, Y, n_epoch=10, n_batch=100, eta=lambda t: 0.1):
        N = X.shape[1]
        t = 0
        for epoch in range(n_epoch):
            print("epoch", epoch + 1)
            for X_batch, Y_batch in generate_batches(X, Y, n_batch):
                self._step(X_batch, Y_batch, eta(t))
                t += 1

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return softmax(X)
