import data_loader
import estimators
import estimator_analysis

Data = data_loader.Data('calibration_p3.pkl')
# Data.test_plot(2)

estimators_arr = [estimators.MaximumValue(), estimators.MinMax(), estimators.MaxBaseline()]

# Estimator = estimators.MaximumValue()
for Estimator in estimators_arr:
    estimator_analyze = estimator_analysis.EstimatorAnalysis(Estimator, Data)
    estimator_analyze.plot_histograms(show=False, save=True)

print('finished')
