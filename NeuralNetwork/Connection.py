import random


class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        #self.weight = 0

    def update_weight(self, rate):
        self.weight += rate * self.downstream_node.delta * self.upstream_node.output
