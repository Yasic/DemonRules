class ConstNode(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.delta = 0
        self.output = 1
        self.upstream = []
        self.downstream = []

    def calculate_hidden_layer_delta(self):
        self.delta = self.output * (1 - self.output) * reduce(
            lambda ret, cnn: ret + cnn.downstream_node.delta * cnn.weight, self.downstream, 0.0)

    def calculate_output(self):
        self.output = 1

    def set_output(self, data):
        self.output = 1

    def append_downstream(self, cnn):
        self.downstream.append(cnn)