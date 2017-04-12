from sigmoid import sigmoid


class Node(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.upstream = []
        self.downstream = []
        self.delta = 0
        self.output = 0

    def set_output(self, output):
        self.output = output

    def append_downstream_connection(self, connection):
        self.downstream.append(connection)

    def append_upstream_connection(self, connection):
        self.upstream.append(connection)

    def calculate_output(self):
        self.output = sigmoid(reduce(lambda result, conn: result + conn.upstream_node.output * conn.weight,
                                     self.upstream, 0.0))

    def calculate_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def calculate_hidden_layer_delta(self):
        self.delta = self.output * (1 - self.output) * reduce(
            lambda result, conn: result + conn.downstream_node.delta * conn.weight, self.downstream, 0.0)