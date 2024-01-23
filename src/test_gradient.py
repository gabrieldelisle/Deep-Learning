import numpy as np
import pytest
from src.layer import BatchNormalizationLayer, DenseLayer
from src.network import Network


def distance_l1(ga, gb):
    return np.max(np.abs(ga - gb) / (np.abs(ga) + np.abs(gb) + 1e-6))


@pytest.mark.parametrize(
    "network",
    [
        # single layer
        (Network([DenseLayer(50, 10)])),
        # multi layers
        (Network([DenseLayer(50, 30), DenseLayer(30, 10)])),
    ],
)
def test_dense_layer_gradient(network):
    dense_layer = network.layers[0]

    X = np.random.random((50, 20))
    Y = np.zeros((10, 20))
    random_i = int(np.random.random() * 10)
    Y[random_i] = 1

    h = 1e-4
    grad_W = np.zeros(dense_layer.W.shape)

    for i in range(dense_layer.W.shape[0]):
        for j in range(dense_layer.W.shape[1]):
            H = np.zeros(dense_layer.W.shape)
            H[i, j] = h

            dense_layer.W += H
            cost_1 = network.cost(X, Y)
            dense_layer.W -= 2 * H
            cost_2 = network.cost(X, Y)
            dense_layer.W += H

            grad_W[i, j] = (cost_1 - cost_2) / 2 / h

    grad_b = np.zeros(dense_layer.b.shape)

    for i in range(dense_layer.b.shape[0]):
        for j in range(dense_layer.b.shape[1]):
            H = np.zeros(dense_layer.b.shape)
            H[i, j] = h

            dense_layer.b += H
            cost_1 = network.cost(X, Y)
            dense_layer.b -= 2 * H
            cost_2 = network.cost(X, Y)
            dense_layer.b += H
            grad_b[i, j] = (cost_1 - cost_2) / 2 / h

    network.step(X, Y, 1)

    assert distance_l1(grad_W, dense_layer.grad_W) < h
    assert distance_l1(grad_b, dense_layer.grad_b) < h

    print(distance_l1(grad_W, dense_layer.grad_W))


@pytest.mark.parametrize(
    "network",
    [
        # multi layers
        (
            Network(
                [DenseLayer(50, 30), BatchNormalizationLayer(30), DenseLayer(30, 10)]
            )
        ),
    ],
)
def test_batch_normalisation_layer_gradient(network):
    batch_normalisation_layer = network.layers[1]

    X = np.random.random((50, 20))
    Y = np.zeros((10, 20))
    random_i = int(np.random.random() * 10)
    Y[random_i] = 1

    network.step(X, Y, 0)

    h = 1e-4
    grad_gamma = np.zeros(batch_normalisation_layer.gamma.shape)

    for i in range(batch_normalisation_layer.gamma.shape[0]):
        for j in range(batch_normalisation_layer.gamma.shape[1]):
            H = np.zeros(batch_normalisation_layer.gamma.shape)
            H[i, j] = h

            batch_normalisation_layer.gamma += H
            cost_1 = network.cost(X, Y)
            batch_normalisation_layer.gamma -= 2 * H
            cost_2 = network.cost(X, Y)
            batch_normalisation_layer.gamma += H

            grad_gamma[i, j] = (cost_1 - cost_2) / 2 / h

    grad_beta = np.zeros(batch_normalisation_layer.beta.shape)

    for i in range(batch_normalisation_layer.beta.shape[0]):
        for j in range(batch_normalisation_layer.beta.shape[1]):
            H = np.zeros(batch_normalisation_layer.beta.shape)
            H[i, j] = h

            batch_normalisation_layer.beta += H
            cost_1 = network.cost(X, Y)
            batch_normalisation_layer.beta -= 2 * H
            cost_2 = network.cost(X, Y)
            batch_normalisation_layer.beta += H
            grad_beta[i, j] = (cost_1 - cost_2) / 2 / h

    assert distance_l1(grad_gamma, batch_normalisation_layer.grad_gamma) < h
    assert distance_l1(grad_beta, batch_normalisation_layer.grad_beta) < h
