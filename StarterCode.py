import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}
rc('font', **font)
# This changes the fonts for all graphs to make them bigger.


def myGauss(x, A, mean, width, base):
    return A*np.exp(-(x-mean)**2/(2*width**2)) + base
# This is my fitting function, a Guassian with a uniform background.

def pulse_shape(t_rise, t_fall):
    xx=np.linspace(0, 4095, 4096)
    yy = -(np.exp(-(xx-1000)/t_rise)-np.exp(-(xx-1000)/t_fall))
    yy[:1000]=0
    yy /= np.max(yy)
    return yy

def fit_pulse(x, A):
    _pulse_template = pulse_shape(20,80)
    xx=np.linspace(0, 4095, 4096)
    return A*np.interp(x, xx, _pulse_template)
# fit_pulse can be used by curve_fit to fit a pulse to the pulse_shape

with open("calibration_p3.pkl","rb") as file:
    calibration_data=pickle.load(file)

pulse_template = pulse_shape(20,80)
plt.plot(pulse_template/2000, label='Pulse Template', color='r')
for itrace in range(10):
    plt.plot(calibration_data['evt_%i'%itrace], alpha=0.3)
plt.xlabel('Sample Index')
plt.ylabel('Readout (V)')
plt.title('Calibration data (10 sets)')
plt.legend(loc=1)
plt.show()
""" 
This shows the first 10 data sets on top of each other.
Always a good idea to look at some of your data before analysing it!
It also plots our pulse template which has been scaled to be slightly 
larger than any of the actual pulses to make it visible.
"""

amp1=np.zeros(1000)
amp2=np.zeros(1000)
area1=np.zeros(1000)
area2=np.zeros(1000)
area3=np.zeros(1000)
pulse_fit=np.zeros(1000)
# These are the 6 energy estimators as empty arrays of the correct size.

for ievt in range(1000):
    current_data = calibration_data['evt_%i'%ievt]
    amp1_calculation = np.max(current_data)
    amp1[ievt] = amp1_calculation
"""
This incorrectly calculates one of the amplitude estimators.
You will want to fix it, and then do the other 5 estimators 
inside this for loop. I.e. you will need to add:
    amp2[ievt] = ...
    area1[ievt] = ...
etc.
"""

amp1*=1000 # convert from V to mV   

num_bins1=40 
bin_range1=(0,0.4) 
"""
These two values were picked by trial and error. You'll 
likely want different values for each estimator.
"""

n1, bin_edges1, _ = plt.hist(amp1, bins=num_bins1, range=bin_range1, color='k', histtype='step', label='Data')
# This plots the histogram AND saves the counts and bin_edges for later use

plt.xlabel('Energy Estimator: Maximum Value (mV)')
plt.ylabel('Events / %2.2f mV'%((bin_range1[-1]-bin_range1[0])/num_bins1));
plt.xlim(bin_range1)  
# If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)

bin_centers1 = 0.5*(bin_edges1[1:]+bin_edges1[:-1])
"""
This gives us the x-data which are the centres of each bin.
This is visually better for plotting errorbars.
More important, it's the correct thing to do for fitting the
Gaussian to our histogram.
It also fixes the shape -- len(n1) < len(bin_edges1) so we
cannot use 
plt.plot(n1, bin_edges1)
as it will give us a shape error.
"""

sig1 = np.sqrt(n1)
sig1=np.where(sig1==0, 1, sig1) 
# The uncertainty on 0 count is 1, not 0. Replace all 0s with 1s.

plt.errorbar(bin_centers1, n1, yerr=sig1, fmt='none', c='k')
# This adds errorbars to the histograms, where each uncertainty is sqrt(y)

popt1, pcov1 = curve_fit(myGauss, bin_centers1, n1, 
             sigma = sig1, p0=(100,0.25,0.05,5), absolute_sigma=True)
n1_fit = myGauss(bin_centers1, *popt1)
"""
n1_fit is our best fit line using our data points.
Note that if you have few enough bins, this best fit
line will have visible bends which look bad, so you
should not plot n1_fit directly. See below.
"""

chisquared1 = np.sum( ((n1 - n1_fit)/sig1 )**2)
dof1 = num_bins1 - len(popt1)
# Number of degrees of freedom is the number of data points less the number of fitted parameters

x_bestfit1 = np.linspace(bin_edges1[0], bin_edges1[-1], 1000)
y_bestfit1 = myGauss(x_bestfit1, *popt1) 
# Best fit line smoothed with 1000 datapoints. Don't use best fit lines with 5 or 10 data points!

fontsize=18
plt.plot(x_bestfit1, y_bestfit1, label='Fit')
plt.text(0.01, 140, r'$\mu$ = %3.2f mV'%(popt1[1]), fontsize=fontsize)
plt.text(0.01, 120, r'$\sigma$ = %3.2f mV'%(popt1[2]), fontsize=fontsize)
plt.text(0.01, 100, r'$\chi^2$/DOF=', fontsize=fontsize)
plt.text(0.01, 80, r'%3.2f/%i'%(chisquared1,dof1), fontsize=fontsize)
plt.text(0.01, 60, r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared1,dof1)), fontsize=fontsize)
plt.legend(loc=1)
plt.show()

"""
Look how bad that chi-squared value (and associated probability) is!
If you look closely, the first 5 data points (on the left) are
responsible for about half of the chi-squared value. It might be
worth excluding them from the fit and subsequent plot.

Now your task is to find the calibration factor which converts the
x-axis of this histogram from mV to keV such that the peak (mu) is 
by definition at 10 keV. You do this by scaling each estimator (i.e.
the values of amp1) by a multiplicative constant with units mV / keV.
Something like:

energy_amp1 = amp1 * conversion_factor1

where you have to find the conversion_factor1 value. Then replot and
refit the histogram using energy_amp1 instead of amp1. 
If you do it correctly, the new mu value will be 10 keV, and the new 
sigma value will be the energy resolution of this energy estimator.

Note: you should show this before/after conversion for your first
energy estimator. To save space, only show the after histograms for
the remaining 5 energy estimators.
"""
