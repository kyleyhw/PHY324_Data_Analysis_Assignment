import numpy as np

class Gaussian(): # height is usually 1; not probability density
    def __init__(self, base, scale, mu, sigma):
        self.scale = scale
        self.mu = mu
        self.sigma = sigma
        self.base = base

    def __call__(self, x):
        result = self.base + self.scale * np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)
        return result


class GaussianZeroCenter(): # height is usually 1; not probability density
    def __init__(self, base, scale, sigma):
        self.scale = scale
        self.sigma = sigma
        self.base = base

    def __call__(self, x):
        result = self.base + self.scale * np.exp(-0.5 * ((x) / self.sigma) ** 2)
        return result