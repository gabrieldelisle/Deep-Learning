import os
import pickle

import numpy as np

DATASET_PATH = "cifar-10-batches-py"


def load_batch(filename):
    with open(os.path.join(DATASET_PATH, filename), "rb") as f:
        dic = pickle.load(f, encoding="bytes")

    X = np.reshape(dic[b"data"], (10000, 3072)).swapaxes(0, 1)
    X = X.astype(float)
    X /= 255.0
    X -= np.mean(X, axis=1).reshape(-1, 1)
    X /= np.std(X, axis=1).reshape(-1, 1)

    y = np.array(dic[b"labels"])
    K = np.max(y)
    Y = np.array(list(map(lambda k: [0] * k + [1] + [0] * (K - k), y))).swapaxes(0, 1)

    return X, Y


def load_training():
    Xs, Ys = [], []
    for i in range(1, 6):
        X, Y = load_batch(f"data_batch_{i}")
        Xs.append(X)
        Ys.append(Y)

    return np.concatenate(Xs, axis=1), np.concatenate(Ys, axis=1)


def accuracy(Y, P):
    return np.mean(np.argmax(P, axis=0) == np.argmax(Y, axis=0))


if __name__ == "__main__":
    from .layer import BatchNormalizationLayer, DenseLayer, ReLuLayer
    from .network import Network

    regularisation_factor = 5e-3
    X_train, Y_train = load_training()
    X_test, Y_test = load_batch("test_batch")

    print(X_train.shape, Y_train.shape)
    model = Network(
        [
            DenseLayer(3072, 50, regularisation_factor),
            BatchNormalizationLayer(50, regularisation_factor),
            ReLuLayer(),
            DenseLayer(50, 10, regularisation_factor),
        ]
    )

    model.fit(X_train, Y_train, 10)
    print(accuracy(Y_test, model.predict(X_test)))
