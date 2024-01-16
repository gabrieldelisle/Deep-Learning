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

    def step(self, X, Y, eta):
        N = X.shape[1]
        P = self.forward(X)
        g = (P - Y) / N
        self.backward(g)

        for layer in self.layers:
            layer.step(eta)

    def fit(self, X, Y, n_epoch=10, n_batch=100, eta=lambda t: 0.1):
        N = X.shape[1]
        t = 0
        for epoch in range(n_epoch):
            print("epoch", epoch + 1)
            for X_batch, Y_batch in generate_batches(X, Y, n_batch):
                self.step(X_batch, Y_batch, eta(t))
                t += 1
            print("cost", self.cost(X, Y))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return softmax(X)

    def backward(self, g):
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def predict(self, X):
        for layer in self.layers:
            X = layer.predict(X)
        return softmax(X)

    def cost(self, X, Y):
        P = self.predict(X)
        cross_entropy_loss = np.log(np.sum(Y * P, axis=0))
        return -np.mean(cross_entropy_loss) + sum(layer.cost() for layer in self.layers)
