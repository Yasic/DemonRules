from Node import *


class Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))

    def set_output(self, datas):
        for i in range(len(datas)):
            self.nodes[i].set_output(datas[i])

    def calculate_output(self):
        for item in self.nodes:
            item.calculate_output()
