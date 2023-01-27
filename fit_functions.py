import numpy as np

class Gaussian(): # height is usually 1; not probability density
    def __init__(self):
        pass

    def __call__(self, x, scale, mu, sigma):
        result = scale * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return result