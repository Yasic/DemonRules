# -*- coding: utf-8 -*-
from NeuralNetwork import *
from Layer import *


def gradient_check(network, sample, label):
    Ed = lambda var1, var2: 0.5 * reduce(lambda a, b: a + b,
                                         map(lambda v: ((v[0] - v[1]) * (v[0] - v[1])), zip(var1, var2)), 0.0)

    network.get_gradient_of_one_sample(sample, label)
    for conn in network.connections:
        actual_gradient = conn.get_gradient()

        epsilon = 0.0001
        conn.weight += epsilon
        error1 = Ed(network.predict(sample), label)

        conn.weight -= 2 * epsilon
        error2 = Ed(network.predict(sample), label)

        expect_gradient = (error2 - error1) / (2 * epsilon)
        print 'expect_gradient: actual_gradient %f:%f' % (expect_gradient, actual_gradient)


if __name__ == '__main__':
    # layers = [Layer(0, 2), Layer(1, 1), Layer(2, 2), Layer(3, 3), Layer(4, 1)]
    layers = [Layer(0, 2), Layer(1, 1), Layer(2, 2), Layer(3, 1)]
    neural_network = NeuralNetwork(layers)
    input_data = [[2, 4], [2, 6], [2, 2], [2, 3], [3, 3], [3, 4], [2, 4], [2, 6], [2, 2], [2, 3], [3, 3], [3, 4], [2, 4], [2, 6], [2, 2], [2, 3], [3, 3], [3, 4]]
    labels = [[6], [8], [4], [5], [6], [7], [6], [8], [4], [5], [6], [7], [6], [8], [4], [5], [6], [7]]
    neural_network.train(input_data, labels, 10, 200)
    neural_network.get_weight()
    neural_network.test([2, 2])
    gradient_check(neural_network, [2, 3], [5])
