import data_loader
import estimators
import estimator_analysis
import signal_estimation

files = ['calibration_p3.pkl', 'noise_p3.pkl', 'signal_p3.pkl']

# Data = data_loader.Data(files[0])
# Data.test_plot(2)

estimators_arr = [estimators.MaximumValue(), estimators.MinMax(), estimators.MaxBaseline(),
                  estimators.SimpleIntegral(), estimators.BaselineSubtractedIntegral(),
                  estimators.LimitedRangeIntegral(), estimators.BaselineSubtractedLimitedRangeIntegral(),
                  estimators.FitToPulseTemplate()]

def run_from_arr():
    Data = data_loader.Data(files[0])

    for Estimator in estimators_arr:
        estimator_analyze = estimator_analysis.EstimatorAnalysis(Estimator, Data)
        estimator_analyze.plot_histograms(show=False, save=True)

    print('finished')

def run_single_estimator():
    Data = data_loader.Data(files[0])

    Estimator = estimators.MaximumValue()
    estimator_analyze = estimator_analysis.EstimatorAnalysis(Estimator, Data)
    estimator_analyze.plot_histograms(show=True, save=True)



def arr_estimate_signal():
    CalibrationData = data_loader.Data(files[0])
    SignalData = data_loader.Data(files[2])

    for Estimator in estimators_arr:
        estimator_analyze = estimator_analysis.EstimatorAnalysis(Estimator, CalibrationData)
        estimator_analyze.plot_histograms(show=False, save=False)

        SignalFitter = signal_estimation.SignalFitter(Estimator, SignalData, estimator_analyze.calibration_factor, estimator_analyze.calibration_factor_error)

        SignalFitter.plot_histogram_without_fit(show=False, save=True)

    print('finished')

def single_estimate_signal():
    Estimator = estimators.BaselineSubtractedLimitedRangeIntegral()
    CalibrationData = data_loader.Data(files[0])
    SignalData = data_loader.Data(files[2])

    estimator_analyze = estimator_analysis.EstimatorAnalysis(Estimator, CalibrationData)
    estimator_analyze.plot_histograms(show=False, save=False)

    SignalFitter = signal_estimation.SignalFitter(Estimator, SignalData, estimator_analyze.calibration_factor, estimator_analyze.calibration_factor_error)

    SignalFitter.plot_histogram_with_fit(show=False, save=True)
    SignalFitter.plot_histogram_with_triple_fit(show=False, save=True)



# run_from_arr()

# run_single_estimator()

arr_estimate_signal()

# single_estimate_signal()
