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
    def __init__(self, Estimator, Data, calibration_factor, calibration_factor_error):
        self.Estimator = Estimator
        self.Data = Data
        self.calibration_factor = calibration_factor
        self.calibration_factor_error = calibration_factor_error
        self.calibration_factor_fractional_error = self.calibration_factor_error / self.calibration_factor

        self.estimator_results = np.zeros(self.Data.length, dtype=float)
        for i in range(self.Data.length):
            self.estimator_results[i] = self.Estimator(self.Data(i))

        self.estimator_results *= calibration_factor

    def plot_histogram_without_fit(self, show=False, save=False):
        fig, ax = plt.subplots(1, 1, figsize = (32, 18))

        self.bin_heights, self.bin_edges, _ = ax.hist(x=self.estimator_results, bins=self.Estimator.number_of_bins, label='data', color='k', histtype='step', range=self.Estimator.range)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        self.count_errors = np.sqrt(self.bin_heights)
        self.count_errors = np.where(self.count_errors == 0, 1, self.count_errors)

        cff.baseplot_errorbars(ax, self.bin_centers, self.bin_heights, yerr=self.count_errors)

        ax.set_title('Histogram of signal data for %s estimator' % self.Estimator.name)
        ax.set_ylabel('events')
        ax.set_xlabel('energy / keV')
        ax.legend(loc='upper left')

        if show:
            fig.show()
        if save:
            fig.savefig('signal_histograms_without_fit/%s_signal_histogram_without_fit.png' % self.Estimator.name)



    def plot_histogram_with_fit(self, show=False, save=False):
        fig, ax = plt.subplots(1, 1, figsize = (32, 18))

        self.bin_heights, self.bin_edges, _ = ax.hist(x=self.estimator_results, bins=self.Estimator.number_of_bins, label='data', color='k', histtype='step', range=self.Estimator.range)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        self.count_errors = np.sqrt(self.bin_heights)
        self.count_errors = np.where(self.count_errors == 0, 1, self.count_errors)

        self.first_peak_bin_heights = self.bin_heights.copy()
        self.first_peak_bin_heights = np.where(self.bin_centers > 3, 0, self.first_peak_bin_heights)

        print(self.first_peak_bin_heights)

        minmax_p0_1 = (0, 200, 5, 1)
        baseline_subtracted_limited_range_integral_p0_1 = (0, 160, 2, 1)
        gaussian_1_p0 = baseline_subtracted_limited_range_integral_p0_1

        gaussian_1 = fit_functions.Gaussian()
        popt1, pcov1 = curve_fit(gaussian_1, xdata=self.bin_centers, ydata=self.first_peak_bin_heights, p0=gaussian_1_p0, sigma=self.count_errors, absolute_sigma=True)
        gaussian_1_base, gaussian_1_scale, gaussian_1_mu, gaussian_1_sigma = tuple(popt1)
        gaussian_1_base_error, gaussian_1_scale_error, gaussian_1_mu_error, gaussian_1_sigma_error = tuple(np.sqrt(np.diag(pcov1)))
        gaussian_1_mu_error += gaussian_1_mu_error * self.calibration_factor_fractional_error
        gaussian_1_sigma_error += gaussian_1_sigma_error * self.calibration_factor_fractional_error
        fitted_gaussian_1 = fitted_functions.Gaussian(*popt1)

        # chi2_1 = cff.calc_raw_chi_squared(self.bin_centers, fitted_gaussian_1(self.bin_centers), np.where(self.first_peak_bin_heights != 0, 0, self.count_errors))
        # dof_1 = cff.calc_dof(self.bin_centers, gaussian_1.num_of_params)
        # chi2_prob_1 = cff.chi2_probability(chi2_1, dof_1)



        self.second_peak_bin_heights = self.bin_heights.copy()
        self.second_peak_bin_heights = np.where(self.bin_centers < 4.5, 0, self.second_peak_bin_heights)
        self.second_peak_bin_heights = np.where(self.bin_centers > 7.5, 0, self.second_peak_bin_heights)

        print(self.second_peak_bin_heights)

        minmax_p0_2 = (0, 40, 7, 0.5)
        baseline_subtracted_limited_range_integral_p0_2 = (0, 60, 7, 1)
        gaussian_2_p0 = baseline_subtracted_limited_range_integral_p0_2

        gaussian_2 = fit_functions.Gaussian()
        popt2, pcov2 = curve_fit(gaussian_2, xdata=self.bin_centers, ydata=self.second_peak_bin_heights, p0=gaussian_2_p0, sigma=self.count_errors, absolute_sigma=True)
        gaussian_2_base, gaussian_2_scale, gaussian_2_mu, gaussian_2_sigma = tuple(popt2)
        gaussian_2_base_error, gaussian_2_scale_error, gaussian_2_mu_error, gaussian_2_sigma_error = tuple(np.sqrt(np.diag(np.abs(pcov2))))
        gaussian_2_mu_error += gaussian_1_mu_error * self.calibration_factor_fractional_error
        gaussian_2_sigma_error += gaussian_1_sigma_error * self.calibration_factor_fractional_error
        fitted_gaussian_2 = fitted_functions.Gaussian(*popt2)

        # chi2_2 = cff.calc_raw_chi_squared(self.bin_centers, fitted_gaussian_2(self.bin_centers), np.where(self.second_peak_bin_heights != 0, 0, self.count_errors))
        # dof_2 = cff.calc_dof(self.bin_centers, gaussian_1.num_of_params)
        # chi2_prob_2 = cff.chi2_probability(chi2_2, dof_2)

        summed_gaussians = Superpose(np.array([fitted_gaussian_1, fitted_gaussian_2]))

        chi2 = cff.calc_raw_chi_squared(self.bin_centers, summed_gaussians(self.bin_centers), self.count_errors)
        dof = cff.calc_dof(self.bin_centers, 8) # hardcoded num of params
        chi2_prob = cff.chi2_probability(chi2, dof)

        cff.baseplot_errorbars(ax, self.bin_centers, self.bin_heights, yerr=self.count_errors)

        x_for_fits = np.linspace(*ax.get_xlim(), 10000)

        # ax.plot(x_for_fits, fitted_gaussian_1(x_for_fits), label='fit1')
        # ax.plot(x_for_fits, fitted_gaussian_2(x_for_fits), label='fit2')
        ax.plot(x_for_fits, summed_gaussians(x_for_fits), label='fit')



        info_sigfigs = 4
        info_fontsize = 22

        information_on_ax = 'number of bins = ' + str(self.Estimator.number_of_bins) + \
                             '\n$\mu_1$ = ' + cff.to_sf(gaussian_1_mu, sf=info_sigfigs) + '$ \pm $' + cff.to_sf(gaussian_1_mu_error, sf=1) + ' keV' + \
                             '\n$\sigma_1$ = ' + cff.to_sf(gaussian_1_sigma, sf=info_sigfigs) + '$ \pm $' + cff.to_sf(gaussian_1_sigma_error, sf=1) + ' keV' + \
                             '\n$\mu_1$ = ' + cff.to_sf(gaussian_2_mu, sf=info_sigfigs) + '$ \pm $' + cff.to_sf(gaussian_2_mu_error, sf=1) + ' keV' + \
                             '\n$\sigma_1$ = ' + cff.to_sf(gaussian_2_sigma, sf=info_sigfigs) + '$ \pm $' + cff.to_sf(gaussian_2_sigma_error, sf=1) + ' keV' + \
                             '\n$\chi^2$ / DOF = ' + cff.to_sf(chi2, sf=info_sigfigs) + ' / ' + str(dof) + ' = ' + cff.to_sf(chi2 / dof, sf=info_sigfigs) + \
                             '\n$\chi^2$ prob = ' + cff.to_sf(chi2_prob, sf=info_sigfigs)


        ax_text = AnchoredText(information_on_ax, loc='upper right', frameon=False, prop=dict(fontsize=info_fontsize))
        ax.add_artist(ax_text)

        ax.set_title('Histogram of signal data for %s estimator' % self.Estimator.name)
        ax.set_ylabel('events')
        ax.set_xlabel('energy / keV')
        ax.legend(loc='upper left')

        if show:
            fig.show()
        if save:
            fig.savefig('signal_histograms_with_fit/%s_signal_histogram_with_fit.png' % self.Estimator.name)


    def plot_histogram_with_triple_fit(self, show=False, save=False):
        fig, ax = plt.subplots(1, 1, figsize = (32, 18))

        self.bin_heights, self.bin_edges, _ = ax.hist(x=self.estimator_results, bins=self.Estimator.number_of_bins, label='data', color='k', histtype='step', range=self.Estimator.range)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        self.count_errors = np.sqrt(self.bin_heights)
        self.count_errors = np.where(self.count_errors == 0, 1, self.count_errors)

        wide_gaussian_p0 = (0, 150, 2)

        wide_gaussian = fit_functions.GaussianZeroCenter()
        popt0, pcov0 = curve_fit(wide_gaussian, xdata=self.bin_centers, ydata=self.first_peak_bin_heights,p0=wide_gaussian_p0, sigma=self.count_errors, absolute_sigma=True)
        wide_gaussian_base, wide_gaussian_scale, wide_gaussian_sigma = tuple(popt0)
        wide_gaussian_base_error, wide_gaussian_scale_error, wide_gaussian_sigma_error = tuple(np.sqrt(np.diag(pcov0)))
        wide_gaussian_mu_error = self.calibration_factor_fractional_error
        wide_gaussian_sigma_error += wide_gaussian_sigma_error * self.calibration_factor_fractional_error
        fitted_wide_gaussian = fitted_functions.GaussianZeroCenter(*popt0)

        self.first_peak_bin_heights = self.bin_heights.copy() - fitted_wide_gaussian(self.bin_heights)
        self.first_peak_bin_heights = np.where(self.bin_centers > 3, 0, self.first_peak_bin_heights)

        print(self.first_peak_bin_heights)

        minmax_p0_1 = (0, 200, 5, 1)
        baseline_subtracted_limited_range_integral_p0_1 = (0, 160, 2, 1)
        gaussian_1_p0 = baseline_subtracted_limited_range_integral_p0_1

        gaussian_1 = fit_functions.Gaussian()
        popt1, pcov1 = curve_fit(gaussian_1, xdata=self.bin_centers, ydata=self.first_peak_bin_heights, p0=gaussian_1_p0, sigma=self.count_errors, absolute_sigma=True)
        gaussian_1_base, gaussian_1_scale, gaussian_1_mu, gaussian_1_sigma = tuple(popt1)
        gaussian_1_base_error, gaussian_1_scale_error, gaussian_1_mu_error, gaussian_1_sigma_error = tuple(np.sqrt(np.diag(pcov1)))
        gaussian_1_mu_error += gaussian_1_mu_error * self.calibration_factor_fractional_error
        gaussian_1_sigma_error += gaussian_1_sigma_error * self.calibration_factor_fractional_error
        fitted_gaussian_1 = fitted_functions.Gaussian(*popt1)

        # chi2_1 = cff.calc_raw_chi_squared(self.bin_centers, fitted_gaussian_1(self.bin_centers), np.where(self.first_peak_bin_heights != 0, 0, self.count_errors))
        # dof_1 = cff.calc_dof(self.bin_centers, gaussian_1.num_of_params)
        # chi2_prob_1 = cff.chi2_probability(chi2_1, dof_1)



        self.second_peak_bin_heights = self.bin_heights.copy() - fitted_wide_gaussian(self.bin_heights)
        self.second_peak_bin_heights = np.where(self.bin_centers < 4.5, 0, self.second_peak_bin_heights)
        self.second_peak_bin_heights = np.where(self.bin_centers > 7.5, 0, self.second_peak_bin_heights)

        print(self.second_peak_bin_heights)

        minmax_p0_2 = (0, 40, 7, 0.5)
        baseline_subtracted_limited_range_integral_p0_2 = (0, 60, 7, 1)
        gaussian_2_p0 = baseline_subtracted_limited_range_integral_p0_2

        gaussian_2 = fit_functions.Gaussian()
        popt2, pcov2 = curve_fit(gaussian_2, xdata=self.bin_centers, ydata=self.second_peak_bin_heights, p0=gaussian_2_p0, sigma=self.count_errors, absolute_sigma=True)
        gaussian_2_base, gaussian_2_scale, gaussian_2_mu, gaussian_2_sigma = tuple(popt2)
        gaussian_2_base_error, gaussian_2_scale_error, gaussian_2_mu_error, gaussian_2_sigma_error = tuple(np.sqrt(np.diag(np.abs(pcov2))))
        gaussian_2_mu_error += gaussian_1_mu_error * self.calibration_factor_fractional_error
        gaussian_2_sigma_error += gaussian_1_sigma_error * self.calibration_factor_fractional_error
        fitted_gaussian_2 = fitted_functions.Gaussian(*popt2)

        # chi2_2 = cff.calc_raw_chi_squared(self.bin_centers, fitted_gaussian_2(self.bin_centers), np.where(self.second_peak_bin_heights != 0, 0, self.count_errors))
        # dof_2 = cff.calc_dof(self.bin_centers, gaussian_1.num_of_params)
        # chi2_prob_2 = cff.chi2_probability(chi2_2, dof_2)

        summed_gaussians = Superpose(np.array([fitted_wide_gaussian, fitted_gaussian_1, fitted_gaussian_2]))

        chi2 = cff.calc_raw_chi_squared(self.bin_centers, summed_gaussians(self.bin_centers), self.count_errors)
        dof = cff.calc_dof(self.bin_centers, 8) # hardcoded num of params
        chi2_prob = cff.chi2_probability(chi2, dof)

        cff.baseplot_errorbars(ax, self.bin_centers, self.bin_heights, yerr=self.count_errors)

        x_for_fits = np.linspace(*ax.get_xlim(), 10000)

        # ax.plot(x_for_fits, fitted_gaussian_1(x_for_fits), label='fit1')
        # ax.plot(x_for_fits, fitted_gaussian_2(x_for_fits), label='fit2')
        ax.plot(x_for_fits, fitted_wide_gaussian(x_for_fits), label='wide fit')
        ax.plot(x_for_fits, summed_gaussians(x_for_fits), label='fit')



        info_sigfigs = 4
        info_fontsize = 22

        information_on_ax = 'number of bins = ' + str(self.Estimator.number_of_bins) + \
                             '\n$\mu_1$ = ' + cff.to_sf(gaussian_1_mu, sf=info_sigfigs) + '$ \pm $' + cff.to_sf(gaussian_1_mu_error, sf=1) + ' keV' + \
                             '\n$\sigma_1$ = ' + cff.to_sf(gaussian_1_sigma, sf=info_sigfigs) + '$ \pm $' + cff.to_sf(gaussian_1_sigma_error, sf=1) + ' keV' + \
                             '\n$\mu_1$ = ' + cff.to_sf(gaussian_2_mu, sf=info_sigfigs) + '$ \pm $' + cff.to_sf(gaussian_2_mu_error, sf=1) + ' keV' + \
                             '\n$\sigma_1$ = ' + cff.to_sf(gaussian_2_sigma, sf=info_sigfigs) + '$ \pm $' + cff.to_sf(gaussian_2_sigma_error, sf=1) + ' keV' + \
                             '\n$\chi^2$ / DOF = ' + cff.to_sf(chi2, sf=info_sigfigs) + ' / ' + str(dof) + ' = ' + cff.to_sf(chi2 / dof, sf=info_sigfigs) + \
                             '\n$\chi^2$ prob = ' + cff.to_sf(chi2_prob, sf=info_sigfigs)


        ax_text = AnchoredText(information_on_ax, loc='upper right', frameon=False, prop=dict(fontsize=info_fontsize))
        ax.add_artist(ax_text)

        ax.set_title('Histogram of signal data for %s estimator' % self.Estimator.name)
        ax.set_ylabel('events')
        ax.set_xlabel('energy / keV')
        ax.legend(loc='upper left')

        if show:
            fig.show()
        if save:
            fig.savefig('signal_histograms_with_fit/%s_signal_histogram_with_triple_fit.png' % self.Estimator.name)
