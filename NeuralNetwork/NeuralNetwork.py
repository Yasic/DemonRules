from Connection import *


class NeuralNetwork(object):
    def __init__(self, layers):
        layer_count = len(layers)
        self.layers = layers
        for i in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[i].nodes
                           for downstream_node in self.layers[i + 1].nodes]
            for cnn in connections:
                cnn.downstream_node.append_upstream(cnn)
                cnn.upstream_node.append_downstream(cnn)

    def train(self, datas, labels, rate, duration):
        for i in range(duration):
            for j in range(len(datas)):
                self.predict(datas[j])
                self.calculate_delta(labels[j])
                self.update_weight(rate)

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calculate_output()

    def calculate_delta(self, label):
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calculate_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calculate_hidden_layer_delta()

    def update_weight(self, rate):
        for layer in self.layers:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def test(self, data):
        self.layers[0].set_output(data)
        for i in range(1, len(self.layers)):
            self.layers[i].calculate_output()
        print self.layers[len(self.layers) - 1].nodes[0].output