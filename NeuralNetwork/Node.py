from Sigmoid import *


class Node(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.delta = 0
        self.output = 0
        self.upstream = []
        self.downstream = []

    def calculate_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def calculate_hidden_layer_delta(self):
        self.delta = self.output * (1 - self.output) * reduce(
            lambda ret, cnn: ret + cnn.downstream_node.delta * cnn.weight, self.downstream, 0.0)

    def calculate_output(self):
        self.output = sigmoid(reduce(lambda ret, cnn: ret + cnn.upstream_node.output * cnn.weight, self.upstream, 0.0))

    def set_output(self, data):
        self.output = data

    def append_upstream(self, cnn):
        self.upstream.append(cnn)

    def append_downstream(self, cnn):
        self.downstream.append(cnn)