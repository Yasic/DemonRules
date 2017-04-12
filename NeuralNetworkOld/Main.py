# -*- coding: utf-8 -*-
from NeuralNetwork import *
from Layer import *

if __name__ == '__main__':
    layers = [Layer(0, 2), Layer(1, 1), Layer(2, 2), Layer(3, 1)]
    neuralNetwork = NeuralNetwork(layers)
    neuralNetwork.train([1, 0, 0, 0], [[1, 1], [1, 0], [0, 0], [0, 1]], 0.1, 10)
    neuralNetwork.predict([1, 1])
