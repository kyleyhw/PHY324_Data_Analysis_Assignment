import numpy as np

class CurveFitFuncs():
    def __init__(self):
        pass

    def baseplot_errorbars(self, ax, x, y, yerr=None, xerr=None, label=None, **kwargs):
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, linestyle='None', capsize=2, label=label, **kwargs)

    def baseplot_errorbars_with_markers(self, ax, x, y, yerr=None, xerr=None, label=None, marker='.', **kwargs):
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, linestyle='None', capsize=2, label=label, marker=marker)
    
    def to_sf(self, num, sf=4):
        return (f'{num:.{sf}g}')



    def residual(self, yarr_measured, yarr_predicted):
        return yarr_measured - yarr_predicted

    def sum_squared_ratio(self, numer, denom):
        return np.sum((numer ** 2) / (denom ** 2))

    def calc_dof(self, yarr_measured, params_in_model):
        dof = len(yarr_measured) - params_in_model
        return dof

    def calc_raw_chi_squared(self, yarr_measured, yarr_predicted, y_uncertainty):
        numer = self.residual(yarr_measured, yarr_predicted)
        denom = y_uncertainty
        return self.sum_squared_ratio(numer, denom)

    def calc_reduced_chi_squared(self, yarr_measured, yarr_predicted, y_uncertainty, params_in_model):
        numer = self.residual(yarr_measured, yarr_predicted)
        denom = y_uncertainty
        dof = len(yarr_measured) - params_in_model
        return self.sum_squared_ratio(numer, denom) / dof