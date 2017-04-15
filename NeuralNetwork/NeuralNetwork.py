from Connection import *
from ConstNode import *


class NeuralNetwork(object):
    def __init__(self, layers):
        layer_count = len(layers)
        self.layers = layers
        self.connections = []
        for i in range(layer_count - 1):
            self.layers[i].nodes.append(ConstNode(i, len(self.layers[i].nodes)))
        connections = []
        for i in range(layer_count - 1):
            for j in range(len(self.layers[i].nodes)):
                for k in range(len(self.layers[i + 1].nodes) - 1):
                    connections.append(Connection(self.layers[i].nodes[j], self.layers[i + 1].nodes[k]))

        for i in range(len(self.layers[layer_count - 2].nodes)):
            for j in range(len(self.layers[layer_count - 1].nodes)):
                connections.append(
                    Connection(self.layers[layer_count - 2].nodes[i], self.layers[layer_count - 1].nodes[j]))
        for cnn in connections:
            cnn.downstream_node.append_upstream(cnn)
            cnn.upstream_node.append_downstream(cnn)
            self.connections.append(cnn)

    def train(self, datas, labels, rate, duration):
        for i in range(duration):
            for j in range(len(datas)):
                self.predict(datas[j])
                self.calculate_delta(labels[j])
                self.update_weight(rate)
                self.get_weight()

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calculate_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

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
        print 'TestOutput: %f' % self.layers[-1].nodes[0].output

    def get_weight(self):
        for item in self.connections:
            print 'weight\t%u-%u:%u-%u:%f' % (
                item.upstream_node.layer_index, item.upstream_node.node_index, item.downstream_node.layer_index,
                item.downstream_node.node_index, item.weight)
        print ''

    def calculate_gradient_of_connections(self):
        for item in self.connections:
            item.calculate_gradient()

    def get_gradient_of_one_sample(self, sample, label):
        self.predict(sample)
        self.calculate_delta(label)
        self.calculate_gradient_of_connections()

