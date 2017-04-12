class Perceptron(object):
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for i in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_data):
        return self.activator(
            reduce(lambda a, b: a + b, map(lambda (x, y): x * y, zip(input_data, self.weights)), 0.0) + self.bias)

    def train(self, input_datas, labels, iteration, rate):
        for item in range(iteration):
            for (input_data, label) in zip(input_datas, labels):
                output = self.predict(input_data)
                self.weights = map(lambda (w, x): w + rate * (label - output) * x, zip(self.weights, input_data))
                self.bias = self.bias + rate * (label - output)
