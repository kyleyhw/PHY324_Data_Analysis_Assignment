import numpy as np
import scipy as sp
from scipy.optimize import curve_fit

class MaximumValue():
    def __init__(self):
        self.name = 'maximum_value'
        self.range = (0, 0.4)
        self.p0 = (0, 100, 0.25, 0.05)
        self.unit = ' mV'

    def __call__(self, data):
        estimate = np.max(data)
        return estimate

class MinMax():
    def __init__(self):
        self.name = 'min_max'
        self.range = None
        self.p0 = (0, 150, 0.3, 0.05)
        self.unit = ' mV'

    def __call__(self, data):
        estimate = np.max(data) - np.min(data)
        return estimate

class MaxBaseline():
    def __init__(self):
        self.name = 'max_baseline'
        self.range = None
        self.p0 = (0, 200, 0.25, 0.05)
        self.unit = ' mV'

    def __call__(self, data):
        baseline = np.mean(data[:1000])
        max = np.max(data)
        estimate = max - baseline
        return estimate



class SimpleIntegral():
    def __init__(self):
        self.name = 'simple_integral'
        self.range = None
        self.p0 = None
        self.unit = ' mV ms'

    def __call__(self, data):
        estimate = sp.integrate.trapz(data, dx=1/4096) # 4 ms = 0.0004s divided into 4096 samples
        return estimate

class BaselineSubtractedIntegral():
    def __init__(self):
        self.name = 'baseline_subtracted_integral'
        self.range = None
        self.p0 = None
        self.unit = ' mV ms'

    def __call__(self, data):
        integral = sp.integrate.trapz(data, dx=1) # 4 ms = 0.0004s divided into 4096 samples
        baseline = np.mean(data[:1000])
        estimate = integral-baseline
        return estimate

class LimitedRangeIntegral():
    def __init__(self):
        self.name = 'limited_range_integral'
        self.range = None # this is the histogram range
        self.p0 = None
        self.unit = ' mV ms'

    def __call__(self, data):
        integration_min = 1000
        integration_max = 1102 # 100us signal width / 4ms * 4096 samples = 102 samples
        integral = sp.integrate.trapz(data[integration_min:integration_max])
        estimate = integral
        return estimate

class BaselineSubtractedLimitedRangeIntegral():
    def __init__(self):
        self.name = 'limited_range_integral'
        self.range = None # this is the histogram range
        self.p0 = None
        self.unit = ' mV ms'

    def __call__(self, data):
        integration_min = 1000
        integration_max = 1102 # 100us signal width / 4ms * 4096 samples = 102 samples
        integral = sp.integrate.trapz(data[integration_min:integration_max])
        baseline = np.mean(data[:1000])
        estimate = integral - baseline
        return estimate



class Template():
    def __init__(self, data, tau_rise, tau_fall, pulse_start):
        self.t = np.arange(len(data))
        results = -(np.exp(-(self.t - pulse_start) / tau_rise) - np.exp(-(self.t - pulse_start) / tau_fall))
        results[:pulse_start] = 0
        results /= np.max(results)
        self.results = results

    def __call__(self, x, A):
        return A * np.interp(x, self.t, self.results)

class FitToGivenTemplate():
    def __init__(self):
        self.name = 'fit_to_pulse_template'
        self.range = None
        self.p0 = None
        self.unit = ' mV'

    def __call__(self, data):
        tau_rise = 20 * 1e-6 * 4096 / (4 * 1e-3)  # 20us * 4096samples / 4ms
        tau_fall = 80 * 1e-6 * 4096 / (4 * 1e-3)  # 80us * 4096samples / 4ms
        pulse_start = 1024 # samples in

        template = Template(data, tau_rise, tau_fall, pulse_start)

        baseline_data = data[:1000]
        error = np.std(baseline_data)
        errors = np.zeros(len(data), dtype=float) + error
        popt, pcov = curve_fit(template, np.arange(0, len(data)), data, sigma=errors, absolute_sigma=True, p0=(2))
        A = popt[0]
        return A

# import matplotlib.pyplot as plt
#
# template = Template(np.arange(4096), 20, 80, 1000)
#
# x = np.arange(10000)
#
# plt.plot(x, template(x, 1))
#
# plt.show()