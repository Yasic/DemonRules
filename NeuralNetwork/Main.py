# -*- coding: utf-8 -*-
from NeuralNetwork import *
from Layer import *

if __name__ == '__main__':
    #layers = [Layer(0, 2), Layer(1, 1), Layer(2, 2), Layer(3, 3), Layer(4, 1)]
    layers = [Layer(0, 2), Layer(1, 1), Layer(2, 1)]
    neuralNetwork = NeuralNetwork(layers)
    neuralNetwork.train([[1, 1], [1, 0], [0, 1], [0, 0]], [[10], [5], [5], [1]], 0.1, 100)
    neuralNetwork.get_weight()
    neuralNetwork.test([1, 1])
