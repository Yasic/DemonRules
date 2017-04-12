from Connection import *


class NeuralNetwork(object):
    def __init__(self, layers):
        self.connections = []
        self.layers = layers
        for i in range(len(layers) - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[i].nodes
                           for downstream_node in self.layers[i + 1].nodes[:-1]]
            for conn in connections:
                self.connections.append(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calculate_output()

    def calculate_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calculate_gradient()

    def calculate_delta(self, label):
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calculate_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calculate_hidden_layer_delta()

    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def train(self, labels, data, rate, iteration):
        for i in range(iteration):
            for d in range(len(data)):
                self.predict(data[d])
                self.calc_delta(labels[d])
                self.update_weight(rate)