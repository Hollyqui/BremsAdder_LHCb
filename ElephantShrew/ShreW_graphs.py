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

# range_value = 12000
# bins=100
# fig, ax = plt.subplots()
# ax.hist(calc_frame_test['TRUE_P'], range = (0,15*range_value), bins=bins, alpha=0.5, color='green', label="True Momentum")
# title = "Distribution of True Momentum"
# ax.legend()
# plt.xlabel("Momentum")
# plt.title(title)
# # plt.yscale("log")
# plt.show()
#
#
# bins=100
# ################## BremAdder
range_value = 2e4
bins=100
fig, ax = plt.subplots()
ax.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=bins, alpha=0.5, color='blue', label="BremAdder")
ba_mse_test = np.average((calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'])**2)
ba_std_test = np.std(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'])
ba_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P']))

# ax.hist(calc_frame_test['eminus_nobrem_P']-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=bins, alpha=0.5, color='green', label="Raw Measurement")
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
#
# range_value = 1.5e5
# fig, ax = plt.subplots()
# ax.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=bins, alpha=0.5, color='blue', label="BremAdder")
# ba_mse_test = np.average((calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'])**2)
# ba_std_test = np.std(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'])
# ba_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P']))
#
# # Fit a normal distribution to the data:
# mu, std = norm.fit(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'])
# data = norm.rvs(mu, std, size=10000)
# # Plot the histogram.
# plt.hist(data, alpha=0.5, color='grey', range = (-range_value,range_value), bins=bins, label="Gaussian of Same STDEV")
#
# title = "Momentum Resolution of BremAdder vs. Normal Distribution"
# ax.legend()
# # plt.yscale("log")
# plt.xlabel("Momentum Resolution")
# plt.title(title)
# plt.show()
#
#
#
# ################ XGBoost
#
xgb_shrew = xgb.XGBRegressor({'nthread':4}) #init model
xgb_shrew.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ShreW.model")
#
xgb_preds_test = xgb_shrew.predict(np.array(x_test))
# range_value = 3e4
# fig, ax = plt.subplots()
# plt.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=bins, alpha=0.5, color='blue', label="BremAdder")
# plt.hist(xgb_preds_test-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=bins, alpha=0.5, color='orange', label="ShreW (XGBoost)")
# xgb_mse_test = np.average((calc_frame_test['TRUE_P']-xgb_preds_test)**2)
# xgb_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-xgb_preds_test))
# xgb_std_test = np.std(calc_frame_test['TRUE_P']-xgb_preds_test)
# # title = "MSE:", xgb_mse_test, "STD:", xgb_std_test
# title = "Momentum Resolution of XGBoost | Standard Deviation: " + str(np.around(xgb_std_test, decimals=1))
# ax.legend()
# plt.xlabel("Momentum Resolution")
# plt.title(title)
# plt.show()
#
# ############### neural net
#
#
# fig, ax = plt.subplots()
nn = pickle.load(open("nn_ShreW", 'rb'))
nn_preds_test = np.array(nn.predict(np.array(x_test)))
# plt.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-30000,30000), bins=bins, alpha=0.5, color='blue', label="BremAdder")
# plt.hist(nn_preds_test-calc_frame_test['TRUE_P'], range = (-30000,30000), bins=bins, alpha=0.5, color='orange', label="ShreW (Neural Net)")
# nn_mse_test = np.average((calc_frame_test['TRUE_P']-nn_preds_test)**2)
# nn_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-nn_preds_test))
# nn_std_test = np.std(calc_frame_test['TRUE_P']-nn_preds_test)
# # title = "MSE:", nn_mse_test, "STD:", nn_std_test
# title = "Momentum Resolution of NN | Standard Deviation: " + str(np.around(nn_std_test, decimals=1))
# ax.legend()
# plt.xlabel("Momentum Resolution")
# plt.title(title)
# plt.show()
#
#
# # fig, ax = plt.subplots()
# # nn2 = pickle.load(open("nn2_ShreW", 'rb'))
# # nn2_preds_test = np.array(nn2.predict(np.array(x_test)))
# # plt.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-30000,30000), bins=50, alpha=0.5, color='blue', label="BremAdder")
# # plt.hist(nn2_preds_test-calc_frame_test['TRUE_P'], range = (-30000,30000), bins=50, alpha=0.5, color='orange', label="ShreW (Neural Net)")
# # nn2_mse_test = np.average((calc_frame_test['TRUE_P']-nn2_preds_test)**2)
# # nn2_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-nn2_preds_test))
# # nn2_std_test = np.std(calc_frame_test['TRUE_P']-nn2_preds_test)
# # # title = "MSE:", nn2_mse_test, "STD:", nn2_std_test
# # title = "Momentum Resolution of nn2 | Standard Deviation: " + str(np.around(nn2_std_test, decimals=1))
# # ax.legend()
# # plt.xlabel("Momentum Resolution")
# # plt.title(title)
# # plt.show()
#
# ######################### GPR
#
# fig, ax = plt.subplots()
# gpr = pickle.load(open("gpr_ShreW", 'rb'))
# gpr_preds_test = np.array(gpr.predict(np.array(x_test)))
# plt.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-30000,30000), bins=bins, alpha=0.5, color='blue', label="BremAdder")
# plt.hist((gpr_preds_test+calc_frame_test['eminus_nobrem_P'])-calc_frame_test['TRUE_P'], range = (-30000,30000), bins=bins, alpha=0.5, color='orange', label="ShreW (Gaussian Process)")
# gpr_mse_test = np.average((calc_frame_test['TRUE_P']-(gpr_preds_test+calc_frame_test['eminus_nobrem_P']))**2)
# gpr_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-(gpr_preds_test+calc_frame_test['eminus_nobrem_P'])))
# gpr_std_test = np.std(calc_frame_test['TRUE_P']-(gpr_preds_test+calc_frame_test['eminus_nobrem_P']))
# # title = "MSE:", gpr_mse_test, "STD:", gpr_std_test
# title = "Momentum Resolution of GPR | Standard Deviation: " + str(np.around(gpr_std_test, decimals=1))
# ax.legend()
# plt.xlabel("Momentum Resolution")
# plt.title(title)
# plt.show()
#
#
# ######################### SVR
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

calc_frame = pd.read_pickle("dataframes/r_calc_frame_notrack")
calc_frame.shape
x_train = calc_frame.drop(['TRUE_P','eminus_P'], axis=1)
svm_y_train = calc_frame['TRUE_P']-calc_frame['eminus_nobrem_P']
scale_x = StandardScaler()
scaled_x_train = scale_x.fit(x_train[:1000])
scaled_x_train = scale_x.transform(x_train)
scaled_x_test = scale_x.transform(x_test)
#test
# fig, ax = plt.subplots()
# svm = pickle.load(open("svr_ShreW", 'rb'))
# svm_preds_test = np.array(svm.predict(np.array(scaled_x_test)))
# plt.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-30000,30000), bins=bins, alpha=0.5, color='blue', label="BremAdder")
# plt.hist((svm_preds_test+calc_frame_test['eminus_nobrem_P'])-calc_frame_test['TRUE_P']-3000, range = (-30000,30000), bins=bins, alpha=0.5, color='orange', label="ShreW (Support Vector Machine)")
# svm_mse_test = np.average((calc_frame_test['TRUE_P']-(svm_preds_test+calc_frame_test['eminus_nobrem_P']))**2)
# svm_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-(svm_preds_test+calc_frame_test['eminus_nobrem_P'])))
# svm_std_test = np.std(calc_frame_test['TRUE_P']-(svm_preds_test+calc_frame_test['eminus_nobrem_P']))
# # title = "MSE:", svm_mse_test, "STD:", svm_std_test
# title = "Momentum Resolution of SVM | Standard Deviation: " + str(np.around(svm_std_test, decimals=1))
# ax.legend()
# plt.xlabel("Momentum Resolution")
# plt.title(title)
# plt.show()
#
# #test
fig, ax = plt.subplots()
svm = pickle.load(open("svr_ShreW", 'rb'))
svm_preds = np.array(svm.predict(np.array(scaled_x_train)))
plt.hist(calc_frame['eminus_P']-calc_frame['TRUE_P'], range = (-30000,30000), bins=bins, alpha=0.5, color='blue', label="BremAdder")
plt.hist((svm_preds+calc_frame['eminus_nobrem_P'])-calc_frame['TRUE_P'], range = (-30000,30000), bins=bins, alpha=0.5, color='orange', label="ShreW (Support Vector Machine)")
svm_mse = np.average((calc_frame['TRUE_P']-(svm_preds-calc_frame['eminus_nobrem_P']))**2)
svm_ae = np.average(np.abs(calc_frame['TRUE_P']-(svm_preds+calc_frame['eminus_nobrem_P'])))
svm_std = np.std(calc_frame['TRUE_P']-(svm_preds+calc_frame['eminus_nobrem_P']))
# title = "MSE:", svm_mse_test, "STD:", svm_std_test
title = "Momentum Resolution of SVM | Standard Deviation: " + str(np.around(svm_std, decimals=1))
ax.legend()
plt.xlabel("Momentum Resolution")
plt.title(title)
plt.show()
#
#
# ################# ensemble_mse_test
# range_value = 1e5
ensemble_preds_test = (nn_preds_test+xgb_preds_test)/2
# fig, ax = plt.subplots()
# plt.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=bins, alpha=0.5, color='blue', label="BremAdder")
# plt.hist(ensemble_preds_test-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=bins, alpha=0.5, color='orange', label="ShreW (Ensemble)")
# ensemble_mse_test = np.average((calc_frame_test['TRUE_P']-ensemble_preds_test)**2)
# ensemble_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-ensemble_preds_test))
# ensemble_std_test = np.std(calc_frame_test['TRUE_P']-ensemble_preds_test)
#
# title = "Momentum Resolution of Ensemble | Standard Deviation: " + str(np.around(ensemble_std_test, decimals=1))
# ax.legend()
# plt.xlabel("Momentum Resolution")
# plt.title(title)
# # plt.yscale("log")
# plt.show()
#
#
# # overlay
# range_value = 3e4
# ensemble_preds_test = (nn_preds_test+xgb_preds_test)/2
# fig, ax = plt.subplots()
# # plt.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=bins, alpha=0.3, color='blue', label="BremAdder")
# plt.hist(ensemble_preds_test-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=bins, alpha=1, color='orange', label="ShreW (Ensemble)")
# plt.hist(nn_preds_test-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=bins, alpha=0.3, color='grey', label="ShreW (Neural Net)")
# plt.hist(xgb_preds_test-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=bins, alpha=0.3, color='grey', label="ShreW (XGBoost)")
#
#
# ensemble_mse_test = np.average((calc_frame_test['TRUE_P']-ensemble_preds_test)**2)
# ensemble_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-ensemble_preds_test))
# ensemble_std_test = np.std(calc_frame_test['TRUE_P']-ensemble_preds_test)
#
# title = "Momentum Resolution of Ensemble | Standard Deviation: " + str(np.around(ensemble_std_test, decimals=1))
# ax.legend()
# plt.xlabel("Momentum Resolution")
# plt.title(title)
# # plt.yscale("log")
# plt.show()
#
#
#
# def run_on_demand():
#     raw_std_test
#     raw_ae_test
#
#     1-(ba_std_test/raw_std_test)
#     1-(ba_ae_test/raw_ae_test)
#     ba_std_test
#     ba_ae_test
#
#     1-(xgb_std_test/raw_std_test)
#     1-(xgb_ae_test/raw_ae_test)
#     xgb_std_test
#     xgb_ae_test
#
#     1-(nn_std_test/raw_std_test)
#     1-(nn_ae_test/raw_ae_test)
#     nn_std_test
#     nn_ae_test
#     1-(gpr_std_test/raw_std_test)
#     1-(gpr_ae_test/raw_ae_test)
#     gpr_std_test
#     gpr_ae_test
#
#
#
#     1-(ensemble_std_test/raw_std_test)
#     1-(ensemble_ae_test/raw_ae_test)
#     ensemble_std_test
#     ensemble_ae_test
#
#
# def run_on_demand():
#     ba_std_test
#     ba_ae_test
#
#     1-(raw_std_test/ba_std_test)
#     1-(raw_ae_test/ba_ae_test)
#     ba_std_test
#     ba_ae_test
#
#     1-(xgb_std_test/ba_std_test)
#     1-(xgb_ae_test/ba_ae_test)
#     xgb_std_test
#     xgb_ae_test
#
#     1-(nn_std_test/ba_std_test)
#     1-(nn_ae_test/ba_ae_test)
#     nn_std_test
#     nn_ae_test
#
#     1-(gpr_std_test/ba_std_test)
#     1-(gpr_ae_test/ba_ae_test)
#     gpr_std_test
#     gpr_ae_test
#
#
#     1-(ensemble_std_test/ba_std_test)
#     1-(ensemble_ae_test/ba_ae_test)
#     ensemble_std_test
#     ensemble_ae_test

#### find 90% region of data
brem_resolution = calc_frame_test['eminus_P']-calc_frame_test['TRUE_P']
ensemble_resolution =ensemble_preds_test-calc_frame_test['TRUE_P']

cut = 0.95
bins=100
temp = sorted(zip(np.abs(brem_resolution), brem_resolution))
trash, brem_sorted = map(list, zip(*temp))
temp = sorted(zip(np.abs(ensemble_resolution), ensemble_resolution))
trash, ensemble_sorted = map(list, zip(*temp))

ens_max = max(ensemble_sorted[0:int(len(ensemble_sorted)*cut)])
brem_max = max(brem_sorted[0:int(len(brem_sorted)*cut)])
ens_min = min(ensemble_sorted[0:int(len(ensemble_sorted)*cut)])
brem_min = min(brem_sorted[0:int(len(brem_sorted)*cut)])

range_value = max(np.abs([ens_max,ens_min,brem_max,brem_min]))
# comment out next line for autoscaled plot
# range_value = 25000
fig, ax = plt.subplots()
ax.hist(brem_sorted[0:int(len(brem_sorted)*cut)], range = (-range_value,range_value), bins=bins, alpha=0.5, color='blue', label="BremAdder")
ax.hist(ensemble_sorted[0:int(len(ensemble_sorted)*cut)], range = (-range_value,range_value), bins=bins, alpha=0.5, color='orange', label="ShreW (Ensemble)")
ax.axvline(x=ens_min, color='orange', linestyle='dashed', linewidth=2)
ax.axvline(x=ens_max, color='orange', linestyle='dashed', linewidth=2)
ax.axvline(x=brem_min, color='b', linestyle='dashed', linewidth=2)
ax.axvline(x=brem_max, color='b', linestyle='dashed', linewidth=2)
title = "Momentum Resolution containing "+str(int(cut*100))+"% of the data"
ax.legend()
plt.xlabel("Momentum Resolution")
# plt.xlim(-range_value,range_value)
plt.title(title)
# plt.yscale("log")
plt.show()



calc_frame_test.columns.tolist()
ph_e = calc_frame_test.iloc[:,1]*calc_frame_test.iloc[:,12]+calc_frame_test.iloc[:,1+14]*calc_frame_test.iloc[:,12+14]+calc_frame_test.iloc[:,1+28]*calc_frame_test.iloc[:,12+28]
np.std(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'])
np.std(calc_frame_test['eminus_nobrem_P']+np.sqrt(ph_e)-calc_frame_test['TRUE_P'])
plt.hist(calc_frame_test['eminus_P']-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=bins, alpha=0.5, color='blue', label="BremAdder")
plt.hist(calc_frame_test['eminus_P']+ph_e-calc_frame_test['TRUE_P'], range = (-range_value,range_value), bins=bins, alpha=0.5, color='orange', label="Algorithmic")

# fig,ax = plt.subplots()
# x = calc_frame_test['TRUE_P']-calc_frame_test['eminus_nobrem_P']
# x = [x for i in zip(x>0, ph_e)]
# np.array(x).shape
# plt.scatter(x,ph_e, s=1)
# plt.plot(np.unique(x), np.poly1d(np.polyfit(x, ph_e, 1))(np.unique(x)))
# plt.ylabel("PT Event")
# plt.xlabel("Momentum Resolution")
# plt.title("Correlation Between Scaled PT and Momentum Resolution")
# plt.xlim(-000,300000)
# plt.ylim(0,12000)

################################ scatter
plt_range = 1000
calc_frame_test = pd.read_pickle("dataframes/test_calc_frame_notrack")


fig, ax = plt.subplots()
plt.plot(calc_frame_test['TRUE_P'][:plt_range], calc_frame_test['TRUE_P'][:plt_range], c='green', label="True Momentum")
ax.scatter(calc_frame_test['TRUE_P'][:plt_range], calc_frame_test['eminus_P'][:plt_range], c='blue', s=1, label="BremAdder")
ax.scatter(calc_frame_test['TRUE_P'][:plt_range], ensemble_preds_test[:plt_range], c='orange', s=1, label="ShreW Ensemble")
ax.legend()
plt.ylabel("Predicted Momentum")
plt.xlabel("True Momentum")
plt.title("Scatter Plot Comparison between ShreW Ensemble and BremAdder")
# plt.yscale("log")
plt.show()


calc_frame_track = pd.read_pickle("dataframes/test_calc_frame")

calc_frame_track.columns.tolist()
