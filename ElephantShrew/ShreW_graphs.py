import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm

calc_frame_test = pd.read_pickle("dataframes/test_calc_frame_notrack")
calc_frame_test.shape

y_test = calc_frame_test['TRUE_P']
x_test = calc_frame_test.drop(['TRUE_P','eminus_P'], axis=1)


################## BremAdder
range_value = 2e4
fig, ax = plt.subplots()
ax.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=100, alpha=0.5, color='blue', label="BremAdder")
ba_mse_test = np.average((calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'])**2)
ba_std_test = np.std(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'])
ba_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P']))

ax.hist(calc_frame_test['eminus_nobrem_P']-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=100, alpha=0.5, color='green', label="Raw Measurement")
raw_mse_test = np.average((calc_frame_test['TRUE_P']-calc_frame_test['eminus_nobrem_P'])**2)
raw_std_test = np.std(calc_frame_test['TRUE_P']-calc_frame_test['eminus_nobrem_P'])
raw_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-calc_frame_test['eminus_nobrem_P']))

# Fit a normal distribution to the data:
# mu, std = norm.fit(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'])
# data = norm.rvs(mu, std, size=10000)
# # Plot the histogram.
# plt.hist(data, alpha=0.5, color='grey', range = (-range_value,range_value), bins=50, label="Gaussian of Same STDEV")

title = "Momentum Resolution of BremAdder vs. Raw Measurement"
ax.legend()
plt.xlabel("Momentum Resolution")
plt.title(title)
# plt.yscale("log")
plt.show()


range_value = 1.5e5
fig, ax = plt.subplots()
ax.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=50, alpha=0.5, color='blue', label="BremAdder")
ba_mse_test = np.average((calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'])**2)
ba_std_test = np.std(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'])
ba_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P']))

# Fit a normal distribution to the data:
mu, std = norm.fit(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'])
data = norm.rvs(mu, std, size=10000)
# Plot the histogram.
plt.hist(data, alpha=0.5, color='grey', range = (-range_value,range_value), bins=50, label="Gaussian of Same STDEV")

title = "Momentum Resolution of BremAdder vs. Normal Distribution"
ax.legend()
plt.xlabel("Momentum Resolution")
plt.title(title)
plt.show()



################ XGBoost

xgb_shrew = xgb.XGBRegressor({'nthread':4}) #init model
xgb_shrew.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ShreW.model")

xgb_preds_test = xgb_shrew.predict(np.array(x_test))
range_value = 3e4
fig, ax = plt.subplots()
plt.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=50, alpha=0.5, color='blue', label="BremAdder")
plt.hist(xgb_preds_test-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=50, alpha=0.5, color='orange', label="ShreW (XGBoost)")
xgb_mse_test = np.average((calc_frame_test['TRUE_P']-xgb_preds_test)**2)
xgb_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-xgb_preds_test))
xgb_std_test = np.std(calc_frame_test['TRUE_P']-xgb_preds_test)
# title = "MSE:", xgb_mse_test, "STD:", xgb_std_test
title = "Momentum Resolution of XGBoost | Standard Deviation: " + str(np.around(xgb_std_test, decimals=1))
ax.legend()
plt.xlabel("Momentum Resolution")
plt.title(title)
plt.show()

############### neural net


fig, ax = plt.subplots()
nn = pickle.load(open("nn_ShreW", 'rb'))
nn_preds_test = np.array(nn.predict(np.array(x_test)))
plt.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-30000,30000), bins=50, alpha=0.5, color='blue', label="BremAdder")
plt.hist(nn_preds_test-calc_frame_test['TRUE_P'], range = (-30000,30000), bins=50, alpha=0.5, color='orange', label="ShreW (Neural Net)")
nn_mse_test = np.average((calc_frame_test['TRUE_P']-nn_preds_test)**2)
nn_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-nn_preds_test))
nn_std_test = np.std(calc_frame_test['TRUE_P']-nn_preds_test)
# title = "MSE:", nn_mse_test, "STD:", nn_std_test
title = "Momentum Resolution of NN | Standard Deviation: " + str(np.around(nn_std_test, decimals=1))
ax.legend()
plt.xlabel("Momentum Resolution")
plt.title(title)
plt.show()



################# ensemble_mse_test
range_value = 1e5
ensemble_preds_test = (nn_preds_test+xgb_preds_test)/2
fig, ax = plt.subplots()
plt.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=500, alpha=0.5, color='blue', label="BremAdder")
plt.hist(ensemble_preds_test-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=500, alpha=0.5, color='orange', label="ShreW (Ensemble)")
ensemble_mse_test = np.average((calc_frame_test['TRUE_P']-ensemble_preds_test)**2)
ensemble_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-ensemble_preds_test))
ensemble_std_test = np.std(calc_frame_test['TRUE_P']-ensemble_preds_test)

title = "Momentum Resolution of Ensemble | Standard Deviation: " + str(np.around(ensemble_std_test, decimals=1))
ax.legend()
plt.xlabel("Momentum Resolution")
plt.title(title)
plt.yscale("log")
plt.show()

raw_std_test
raw_ae_test

1-(ba_std_test/raw_std_test)
1-(ba_ae_test/raw_ae_test)
ba_std_test
ba_ae_test

1-(xgb_std_test/raw_std_test)
1-(xgb_ae_test/raw_ae_test)
xgb_std_test
xgb_ae_test

1-(nn_std_test/raw_std_test)
1-(nn_ae_test/raw_ae_test)
nn_std_test
nn_ae_test


1-(ensemble_std_test/raw_std_test)
1-(ensemble_ae_test/raw_ae_test)
ensemble_std_test
ensemble_ae_test


#### find 90% region of data
brem_resolution = calc_frame_test['eminus_P']-calc_frame_test['TRUE_P']
ensemble_resolution =ensemble_preds_test-calc_frame_test['TRUE_P']

cut = 0.99
# ens_start = int((len(ensemble_resolution)/2-len(ensemble_resolution)*cut/2))
# ens_end = int(ens_start+len(ensemble_resolution)*cut)
# brems_start = int((len(brem_resolution)/2-len(brem_resolution)*cut/2))
# brems_end = int(brems_start+len(brem_resolution)*cut)

temp = sorted(zip(np.abs(brem_resolution), brem_resolution))
trash, brem_sorted = map(list, zip(*temp))
temp = sorted(zip(np.abs(ensemble_resolution), ensemble_resolution))
trash, ensemble_sorted = map(list, zip(*temp))

ens_max = max(ensemble_sorted[0:int(len(ensemble_sorted)*cut)])
brem_max = max(brem_sorted[0:int(len(brem_sorted)*cut)])
ens_min = min(ensemble_sorted[0:int(len(ensemble_sorted)*cut)])
brem_min = min(brem_sorted[0:int(len(brem_sorted)*cut)])

range_value = 3e4
plt.hist(brem_sorted[0:int(len(brem_sorted)*cut)], range = (-range_value,range_value), bins=300, alpha=0.5, color='blue', label="BremAdder")
plt.hist(ensemble_sorted[0:int(len(ensemble_sorted)*cut)], range = (-range_value,range_value), bins=300, alpha=0.5, color='orange', label="ShreW (Ensemble)")
plt.axvline(x=ens_min, color='orange', linestyle='dashed', linewidth=2)
plt.axvline(x=ens_max, color='orange', linestyle='dashed', linewidth=2)
plt.axvline(x=brem_min, color='b', linestyle='dashed', linewidth=2)
plt.axvline(x=brem_max, color='b', linestyle='dashed', linewidth=2)
