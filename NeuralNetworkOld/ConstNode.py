class ConstNode(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.upstream = []
        self.downstream = []
        self.delta = 0
        self.output = 1

    def append_downstream_connection(self, connection):
        self.downstream.append(connection)

    def calculate_hidden_layer_delta(self):
        self.delta = self.output * (1 - self.output) * reduce(
            lambda result, conn: result + conn.downstream_node.delta * conn.weight, self.downstream, 0.0)