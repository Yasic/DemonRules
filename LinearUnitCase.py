# -*- coding: utf-8 -*-
from LinearUnit import LinearUnit


def get_training_data():
    input_datas = [[3], [5], [1], [4], [5]]
    labels = [20000, 30000, 11500, 22000, 28000]
    return input_datas, labels


def train_linear_unit():
    lu = LinearUnit(1)
    (input_datas, labels) = get_training_data()
    lu.train(input_datas, labels, 10, 0.01)
    return lu

if __name__ == '__main__':
    linear_unit = train_linear_unit()
    print linear_unit
    print 'Work 3.4 years, yearly salary = %.2f' % linear_unit.predict([3.4])
    print 'Work 15 years, yearly salary = %.2f' % linear_unit.predict([15])
    print 'Work 1.5 years, yearly salary = %.2f' % linear_unit.predict([1.5])
    print 'Work 6.3 years, yearly salary = %.2f' % linear_unit.predict([6.3])