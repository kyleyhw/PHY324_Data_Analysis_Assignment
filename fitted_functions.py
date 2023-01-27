import numpy as np

class Gaussian(): # height is usually 1; not probability density
    def __init__(self, scale, mu, sigma):
        self.scale = scale
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        result = self.scale * np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)
        return result