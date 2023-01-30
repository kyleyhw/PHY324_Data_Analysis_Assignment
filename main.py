import data_loader
import estimators
import estimator_analysis

Data = data_loader.Data('calibration_p3.pkl')
# Data.test_plot(2)

def run_from_arr():
    estimators_arr = [estimators.MaximumValue(), estimators.MinMax(), estimators.MaxBaseline(),
                      estimators.SimpleIntegral(), estimators.BaselineSubtractedIntegral(),
                      estimators.LimitedRangeIntegral(), estimators.BaselineSubtractedLimitedRangeIntegral(),
                      estimators.FitToGivenTemplate()]

    for Estimator in estimators_arr:
        estimator_analyze = estimator_analysis.EstimatorAnalysis(Estimator, Data)
        estimator_analyze.plot_histograms(show=False, save=True)

    print('finished')

def run_single_estimator():
    Estimator = estimators.FitToGivenTemplate()
    estimator_analyze = estimator_analysis.EstimatorAnalysis(Estimator, Data)
    estimator_analyze.plot_histograms(show=True, save=True)

run_from_arr()

# run_single_estimator()
