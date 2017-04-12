# -*- coding: utf-8 -*-
from NeuralNetwork import *
from Layer import *

if __name__ == '__main__':
    layers = [Layer(0, 2), Layer(1, 1), Layer(2, 2), Layer(3, 1)]
    neuralNetwork = NeuralNetwork(layers)
    neuralNetwork.train([[1, 21], [1, 22], [3, 23], [5, 27]], [[11500], [11000], [18000], [25000]], 0.1, 10)
    neuralNetwork.test([1, 21])
