import numpy as np

class Gaussian(): # height is usually 1; not probability density
    def __init__(self):
        self.num_of_params = 4
        pass

    def __call__(self, x, base, scale, mu, sigma):
        result = base + scale * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return result