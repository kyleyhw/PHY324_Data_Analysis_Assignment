import numpy as np

class MinMax():
    def __init__(self, data):
        self.data = data

    def __call__(self):
        estimate = np.max(self.data) - np.min(self.data)
        return estimate

class