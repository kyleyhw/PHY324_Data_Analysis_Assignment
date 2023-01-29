import numpy as np

class MaximumValue():
    def __init__(self):
        self.name = 'Maximumvalue'

    def __call__(self, data):
        estimate = np.max(data)
        return estimate


class MinMax():
    def __init__(self):
        self.name = 'MinMax'

    def __call__(self, data):
        estimate = np.max(data) - np.min(data)
        return estimate

class MaxBaseline():
    def __init__(self):
        self.name = 'MaxBaseline'