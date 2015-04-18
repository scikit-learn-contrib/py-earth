import os
import numpy


class BaseContainer(object):
    filename = os.path.join(os.path.dirname(__file__), '../test_data.csv')
    data = numpy.genfromtxt(filename, delimiter=',', skip_header=1)
    X = numpy.array(data[:, 0:5])
    y = numpy.array(data[:, 5])
