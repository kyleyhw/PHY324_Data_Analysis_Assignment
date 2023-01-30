import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}
rc('font', **font)
from matplotlib.offsetbox import AnchoredText
from scipy.optimize import curve_fit

import fit_functions
import fitted_functions
from curve_fit_funcs import CurveFitFuncs

cff = CurveFitFuncs()

class Superpose():
    def __init__(self, callables_array):
        self.callables_array = callables_array

    def __call__(self, x):
        return np.sum(np.array([func(x) for func in self.callables_array]), axis=0)



class SignalFitter():
    def __init__(self, Estimator, Data):
        self.Estimator = Estimator
        self.Data = Data

        self.estimator_results = np.zeros(self.Data.length, dtype=float)
        for i in range(self.Data.length):
            self.estimator_results[i] = self.Estimator(self.Data(i))

    def plot_histogram(self, show=False, save=False):
        fig, ax = plt.subplots(1, 1, figsize = (32, 18))

        self.bin_heights, self.bin_edges, _ = ax.hist(x=self.estimator_results, bins=self.Estimator.number_of_bins, label='data', color='k', histtype='step', range=self.Estimator.range)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        self.count_errors = np.sqrt(self.bin_heights)
        self.count_errors = np.where(self.count_errors == 0, 1, self.count_errors)

        ax.set_title('Histogram of signal data for %s estimator' % self.Estimator.name)



        if show:
            fig.show()
        if save:
            fig.savefig('signal_histograms/%s_signal_histogram.png' % self.Estimator.name)