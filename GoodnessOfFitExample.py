# -*- coding: utf-8 -*-
"""
This code does a basic Goodness of Fit analysis of some made up data.

First, it creates the data. The data come from a quadratic function, and
then some noise is added. The noise is Gaussian, with a mean of zero and
a standard deviation of "noise_level". You should play with the noise_level
variable to see what happens with data with more/less noise. Values between 
0.1 and 10 are instructive. You can also play with the "num_bins" variable
which determines how many data points you have. As usual, statistical 
analyses are more reliable with more data points.

Next, the code fits the noisy data to a quadratic function (which is by 
definition optimal) and a Gaussian function. The data were carefully chosen 
so that the Gaussian fit will often be decent. Then it plots the noisy data
along with the two best fit lines.

Next, it calculates the residuals (data - best_fit) for both fits. It 
squares each residual and divides it by the square of its y-uncertainty.
The sum of this is the chi-squared statistic. Loosely speaking, this
measures the variance of the data with respect to the fitted line, scaled
by the uncertainties. If everything goes well, the average value should be
around 1 as the average data point *should* miss the best-fit line by about
one errorbar. However, you don't want the actual average (divide by N,
where N is the number of data points, i.e. the "num_bins" variable). This
is because each fitting parameter reduces the number of degrees of freedom
by, in essence, promising to perfectly fit one data point (each) regardless
of what the data looks like. So we divide by (N - X) where X is the number
of fitted parameters. This is called the reduced chi squared.

The reduced chi squared should be approximately equal to 1. Much higher 
than 1 and your function is not a good fit (you need a better theory). Much 
lower than 1 and your uncertainties are so large that you could probably 
fit many different functions with equal success. In this latter case, it is 
like you are drawing the best fit line by hand, but instead of using a 
mechanical pencil to make a thin line you are using a large Sharpie, making 
a super thick line. The thick line will easily cover up most of your data 
points no matter what shape (function) your use to draw the line.

Finally, it uses the scipy.stats.chi2.cdf function (cumulative distribution
function) to determine the probability that the best-fit function is good.
It estimates how likely your number of data points, if truly drawn from 
your best-fit function, would look this scattered or more. If that 
probability is high, your function is likely valid. If low, you should try
a different function (or double check your uncertainties). Generally, a
5% or lower probability is questionable, and anything below 1% is safe to
say something went wrong. If the probability is "too high" (like 95% or
higher) you might want to check whether your uncertainties are too large 
as you may be "fitting with a Sharpie".

Because this is random, you can run the program a few times and see wildly
different results. Sometimes (but not often) the Gaussian fit is actually 
better, according to this test. So don't rely too heavily on it. Also, it's
always wise to specifically look at your reduced chi squared as well as 
the probability. Finally, you should never trust any statistical test
without eyeballing the data. Always plot it and look at it to make sure
things are reasonable.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.stats as stats


def myGauss(x, A, mean, sigma):
    return A*np.exp( - (x - mean)**2/(2*sigma**2) )

def quad(x, A, mid_x, shape):
    return A - shape*(x - mid_x)**2

num_bins = 100
p0 = (25, 5, 1)
xdata = np.linspace(0, 10, num_bins)
ydata = quad(xdata, *p0)

noise_level = 2
noise = np.random.normal(0, noise_level, num_bins)
noisy_ydata = ydata + noise

GaussGuess = (25, 5, 3)

popt1, pcov1 = optimize.curve_fit(quad, xdata, noisy_ydata, p0=p0)
popt2, pcov2 = optimize.curve_fit(myGauss, xdata, noisy_ydata, p0=GaussGuess)

quad_theory = quad(xdata, *popt1)
Gauss_theory = myGauss(xdata, *popt2)

plt.errorbar(xdata, noisy_ydata, yerr=noise_level, fmt=".")
plt.plot(xdata, quad_theory, color="red", label="Quadratic Fit")
plt.plot(xdata, Gauss_theory, color="purple", label="Gaussian Fit")
plt.legend()
plt.show()

R1 = noisy_ydata - quad_theory
R2 = noisy_ydata - Gauss_theory

quad_chi2 = sum( (R1/noise_level)**2 )
Gauss_chi2 = sum( (R2/noise_level)**2 )

quad_df = num_bins - len(popt1)
Gauss_df = num_bins - len(popt2)

print("Quadratic fit: Chi-squared:", quad_chi2)
print("Deg. Freedom:", quad_df, "Reduced Chi-squared:", quad_chi2/quad_df)
print("Probability it's a good fit:", 1 - stats.chi2.cdf(quad_chi2, quad_df) )

print("Gaussian fit: Chi-squared:", Gauss_chi2)
print("Deg. Freedom:", Gauss_df, "Reduced Chi-squared:", Gauss_chi2/Gauss_df)
print("Probability it's a good fit:", 1 - stats.chi2.cdf(Gauss_chi2, Gauss_df) )
