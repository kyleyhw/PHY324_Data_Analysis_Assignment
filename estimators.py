import numpy as np

class MaximumValue():
    def __init__(self):
        self.name = 'Maximumvalue'
        self.range = (0, 0.4)
        self.p0 = (0, 100, 0.25, 0.05)

    def __call__(self, data):
        estimate = np.max(data)
        return estimate


class MinMax():
    def __init__(self):
        self.name = 'MinMax'
        self.range = None
        self.p0 = (0, 150, 0.3, 0.05)

    def __call__(self, data):
        estimate = np.max(data) - np.min(data)
        return estimate

class MaxBaseline():
    def __init__(self):
        self.name = 'MaxBaseline'
        self.range = None
        self.p0 = None

    def __call__(self, data):
        baseline = np.mean(data[:1000])
        max = np.max(data)
        estimate = max - baseline
        return estimate