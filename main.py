import data_loader
import estimators
import estimator_analysis

Data = data_loader.Data('calibration_p3.pkl')
# Data.test_plot(2)

Estimator = estimators.MaximumValue()

estimator_analysis = estimator_analysis.EstimatorAnalysis(Estimator, Data)

estimator_analysis.plot_histograms(show=True, save=True)