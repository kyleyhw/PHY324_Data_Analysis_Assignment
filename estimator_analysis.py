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



class EstimatorAnalysis():
    def __init__(self, Estimator, Data):
        self.Data = Data
        self.Estimator = Estimator

        self.estimator_results = np.zeros(self.Data.length, dtype=float)
        for i in range(self.Data.length):
            self.estimator_results[i] = self.Estimator(self.Data(i))

    def plot_histograms(self, show=False, save=False):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(32, 18))

        self.uncalibrated_bin_heights, self.uncalibrated_bin_edges, _ = ax1.hist(x=self.estimator_results, bins=self.Estimator.number_of_bins, label='data', color='k', histtype='step', range=self.Estimator.range) # HARDCODED RANGE
        self.uncalibrated_bin_centers = (self.uncalibrated_bin_edges[:-1] + self.uncalibrated_bin_edges[1:]) / 2

        self.count_errors = np.sqrt(self.uncalibrated_bin_heights)
        self.count_errors = np.where(self.count_errors == 0, 1, self.count_errors)

        fit_function_uncalibrated = fit_functions.Gaussian()

        popt, pcov = curve_fit(fit_function_uncalibrated, self.uncalibrated_bin_centers, self.uncalibrated_bin_heights, sigma=self.count_errors, absolute_sigma=True, p0=self.Estimator.p0) # HARDCODED P0 INITIAL GUESS
        self.error_in_uncalibrated_fit_parameters = np.sqrt(np.diag(pcov))

        (self.uncalibrated_gaussian_base, self.uncalibrated_gaussian_scale, self.uncalibrated_gaussian_mu, self.uncalibrated_gaussian_sigma) = tuple(popt)
        (self.uncalibrated_gaussian_base_error, self.uncalibrated_gaussian_scale_error, self.uncalibrated_gaussian_mu_error, self.uncalibrated_gaussian_sigma_error) = tuple(self.error_in_uncalibrated_fit_parameters)

        fitted_function_uncalibrated = fitted_functions.Gaussian(*popt)


        self.calibration_energy = 10 # keV
        self.calibration_factor = self.calibration_energy / self.uncalibrated_gaussian_mu
        self.calibration_factor_error = (self.uncalibrated_gaussian_mu_error/self.uncalibrated_gaussian_mu) * self.calibration_factor
        self.calibrated_estimator_results = self.estimator_results * self.calibration_factor

        self.calibrated_bin_heights, self.calibrated_bin_edges, _ = ax2.hist(x=self.calibrated_estimator_results, bins=self.Estimator.number_of_bins, label='data', color='k', histtype='step')
        self.calibrated_bin_centers = (self.calibrated_bin_edges[:-1] + self.calibrated_bin_edges[1:]) / 2

        fit_function_calibrated = fit_functions.Gaussian()

        if self.Estimator.p0 != None:
            calibrated_p0 = tuple(value * scale_factor for value, scale_factor in zip(self.Estimator.p0, (1, 1, self.calibration_factor, self.calibration_factor)))
        else:
            calibrated_p0 = None

        popt, pcov = curve_fit(fit_function_calibrated, self.calibrated_bin_centers, self.calibrated_bin_heights, sigma=self.count_errors, absolute_sigma=True, p0=calibrated_p0)
        self.error_in_calibrated_fit_parameters = np.sqrt(np.diag(pcov))

        (self.calibrated_gaussian_base, self.calibrated_gaussian_scale, self.calibrated_gaussian_mu, self.calibrated_gaussian_sigma) = tuple(popt)
        (self.calibrated_gaussian_base_error, self.calibrated_gaussian_scale_error, self.calibrated_gaussian_mu_error, self.calibrated_gaussian_sigma_error) = tuple(self.error_in_calibrated_fit_parameters)

        fitted_function_calibrated = fitted_functions.Gaussian(*popt)




        self.predicted_uncalibrated_bin_heights = fitted_function_uncalibrated(self.uncalibrated_bin_centers)
        self.uncalibrated_raw_chi_squared = cff.calc_raw_chi_squared(self.uncalibrated_bin_heights, self.predicted_uncalibrated_bin_heights, self.count_errors)
        self.uncalibrated_dof = cff.calc_dof(self.uncalibrated_bin_heights, fit_function_uncalibrated.num_of_params)
        self.uncalibrated_reduced_chi_squared = self.uncalibrated_raw_chi_squared / self.uncalibrated_dof
        self.uncalibrated_chi2_prob = cff.chi2_probability(self.uncalibrated_raw_chi_squared, self.uncalibrated_dof)

        self.predicted_calibrated_bin_heights = fitted_function_calibrated(self.calibrated_bin_centers)
        self.calibrated_raw_chi_squared = cff.calc_raw_chi_squared(self.calibrated_bin_heights, self.predicted_calibrated_bin_heights, self.count_errors)
        self.calibrated_dof = cff.calc_dof(self.calibrated_bin_heights, fit_function_calibrated.num_of_params)
        self.calibrated_reduced_chi_squared = self.calibrated_raw_chi_squared / self.calibrated_dof
        self.calibrated_chi2_prob = cff.chi2_probability(self.calibrated_raw_chi_squared, self.calibrated_dof)

        cff.baseplot_errorbars(ax1, self.uncalibrated_bin_centers, self.uncalibrated_bin_heights, yerr=self.count_errors)
        cff.baseplot_errorbars(ax2, self.calibrated_bin_centers, self.calibrated_bin_heights, yerr=self.count_errors)

        x_for_plotting_uncalibrated = np.linspace(*ax1.get_xlim(), 10000)
        x_for_plotting_calibrated = np.linspace(*ax2.get_xlim(), 10000)

        ax1.plot(x_for_plotting_uncalibrated, fitted_function_uncalibrated(x_for_plotting_uncalibrated), label='fit')
        ax2.plot(x_for_plotting_calibrated, fitted_function_calibrated(x_for_plotting_calibrated), label='fit')

        info_sigfigs = 4
        info_fontsize = 22

        ax1.set_ylabel('events')
        ax1.set_xlabel('detector response amplitude / mV')
        ax1.set_title('Uncalibrated data with Gaussian fit for %s estimator' % self.Estimator.name)
        ax1.legend(loc='upper right')

        information_on_ax1 = 'number of bins = ' + str(self.Estimator.number_of_bins) + \
                             '\n$\mu$ = ' + cff.to_sf(self.uncalibrated_gaussian_mu, sf=info_sigfigs) + '$ \pm $' + cff.to_sf(self.uncalibrated_gaussian_mu_error, sf=1) + self.Estimator.unit + \
                             '\n$\sigma$ = ' + cff.to_sf(self.uncalibrated_gaussian_sigma, sf=info_sigfigs) + '$ \pm $' + cff.to_sf(self.uncalibrated_gaussian_sigma_error, sf=1) + self.Estimator.unit + \
                             '\n$\chi^2$ / DOF = ' + cff.to_sf(self.uncalibrated_raw_chi_squared, sf=info_sigfigs) + ' / ' + str(self.uncalibrated_dof) + ' = ' + cff.to_sf(self.uncalibrated_reduced_chi_squared, sf=info_sigfigs) + \
                             '\n$\chi^2$ prob = ' + cff.to_sf(self.uncalibrated_chi2_prob, sf=info_sigfigs)

        ax1_text = AnchoredText(information_on_ax1, loc='upper left', frameon=False, prop=dict(fontsize=info_fontsize))
        ax1.add_artist(ax1_text)


        ax2.set_ylabel('events')
        ax2.set_xlabel('particle energy / keV')
        ax2.set_title('Calibrated data with Gaussian fit for %s estimator' % self.Estimator.name)
        ax2.legend(loc='upper right')

        information_on_ax2 = 'number of bins = ' + str(self.Estimator.number_of_bins) + \
                             '\ncalibration factor = ' + cff.to_sf(self.calibration_factor, sf=info_sigfigs) + '$ \pm $' + cff.to_sf(self.calibration_factor_error, sf=1) + '%f (%s ) / mV' % (self.calibration_factor, self.Estimator.unit) + \
                             '\n$\mu_c$ = ' + cff.to_sf(self.calibrated_gaussian_mu, sf=info_sigfigs) + '$ \pm $' + cff.to_sf(self.calibrated_gaussian_mu_error, sf=1) + ' keV' + \
                             '\n$\sigma_c$ = ' + cff.to_sf(np.abs(self.calibrated_gaussian_sigma), sf=info_sigfigs) + '$ \pm $' + cff.to_sf(self.uncalibrated_gaussian_sigma_error, sf=1) + ' keV' + \
                             '\n$\chi^2$ / DOF = ' + cff.to_sf(self.calibrated_raw_chi_squared, sf=info_sigfigs) + ' / ' + str(self.calibrated_dof) + ' = ' + cff.to_sf(self.calibrated_reduced_chi_squared, sf=info_sigfigs) + \
                             '\n$\chi^2$ prob = ' + cff.to_sf(self.calibrated_chi2_prob, sf=info_sigfigs)

        ax2_text = AnchoredText(information_on_ax2, loc='upper left', frameon=False, prop=dict(fontsize=info_fontsize))
        ax2.add_artist(ax2_text)


        # print('Estimator %s' % self.Estimator.name)
        # print('Calibration factor = %f $\pm$ %f / (%s / mV )' % (self.calibration_factor, self.calibration_factor_error, self.Estimator.unit))
        # print('Energy resolution = %f $\pm$ %f keV' % (self.calibrated_gaussian_sigma, self.calibrated_gaussian_sigma_error))
        # print('Fit $\chi^2$ probability %f' %self.calibrated_chi2_prob)

        print('%s & $ %f \pm %f  / (\%s / \mV ) $ & $%f \pm %f \keV $ & $ %f $' % (self.Estimator.name, self.calibration_factor, self.calibration_factor_error, self.Estimator.unit, self.calibrated_gaussian_sigma, self.calibrated_gaussian_sigma_error, self.calibrated_chi2_prob))

        # print()
        # print()
        # print()

        if show:
            fig.show()

        if save:
            fig.savefig('estimator_histograms/%s_histograms.png'%self.Estimator.name)

