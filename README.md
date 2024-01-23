# Deep-Learning

Implementation of a multi layer neural network, which can use layers of types 
- Dense
- ReLu
- Bacth Normalisation

## Usage

This implementation is tested in `main.py` against [CIFAR 10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. This dataset contains 60 000 pictures labeled within ten categories.
The following architecture was used : Dense -> ReLu -> BatchNorm -> Dense -> Softmax.
When training this neural network with the 50 000 first pictures of CIFAR and testing on the last 10 000, the neural network gets an accuracy of 48% 