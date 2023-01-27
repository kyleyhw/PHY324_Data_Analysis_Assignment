import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import fit_functions
import fitted_functions
from scipy.optimize import curve_fit

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}
rc('font', **font)

class EstimatorAnalysis():
    def __init__(self, Estimator, Data):
        self.Data = Data
        self.Estimator = Estimator
        self.estimator_results = np.zeros(self.Data.length, dtype=float)

        for i in range(self.Data.length):
            self.estimator_results[i] = self.Estimator(self.Data(i))

    def plot_histograms(self, show=False, save=False):

        fig, (ax1, ax2) = plt.subplots(2, 1)
        self.number_of_bins = 100

        self.uncalibrated_bin_heights, self.uncalibrated_bin_edges, _ = plt.hist(x=self.estimator_results, bins=self.number_of_bins, label='data')
        self.uncalibrated_bin_centers = 0.5 * (self.uncalibrated_bin_edges[:-1] + self.uncalibrated_bin_edges[1:])
        self.uncalibrated_errors = np.sqrt(self.estimator_results)
        self.uncalibrated_errors = np.where(self.uncalibrated_errors == 0, 1, self.uncalibrated_errors)

        fit_function = fit_functions.Gaussian()

        popt, pcov = curve_fit(fit_function, self.uncalibrated_bin_centers, self.uncalibrated_bin_heights)

        fitted_function = fitted_functions.Gaussian(*popt)

        self.calibrated_bin_heights, self.calibrated_bin_edges =


