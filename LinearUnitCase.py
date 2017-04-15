# -*- coding: utf-8 -*-
from LinearUnit import LinearUnit


def get_training_data():
    input_datas = [[1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4]]
    labels = [1, 3, 4, 5, 6, 7]
    return input_datas, labels


def train_linear_unit():
    lu = LinearUnit(2)
    (input_datas, labels) = get_training_data()
    lu.train(input_datas, labels, 100, 0.01)
    return lu

if __name__ == '__main__':
    linear_unit = train_linear_unit()
    print linear_unit
    print '%d' % linear_unit.predict([4, 5])
    #print 'Work 3.4 years, yearly salary = %.2f' % linear_unit.predict([3.4])
    #print 'Work 15 years, yearly salary = %.2f' % linear_unit.predict([15])
    #print 'Work 1.5 years, yearly salary = %.2f' % linear_unit.predict([1.5])
    #print 'Work 6.3 years, yearly salary = %.2f' % linear_unit.predict([6.3])