from dataloader import Data_loader
from mathematics import Mathematics
import pandas as pd
import numpy as np
from data_handler import Data_handler
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pickle

#%%
############## PREPARE DATASET ########################

electron_usable = pd.read_pickle("dataframes/r_electron_usable_down")
class_frame = pd.read_pickle("dataframes/r_class_frame_down")
calc_frame = pd.read_pickle("dataframes/r_calc_frame_notrack")
ElePhanT = xgb.XGBClassifier({'nthread':4}) #init model
ElePhanT.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ElePhant_Classifier_MT.model")

filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set_down"
df = Data_loader.load(filename, 50000, 100000)

handler = Data_handler(df)
predicted_frame = handler.assign_prediction(electron_usable, ElePhanT, class_frame)
calc_frame = handler.calc_frame(predicted_frame, n_cand=3)
calc_frame.columns.tolist()



calc_frame.to_pickle("dataframes/r_calc_frame_notrack")

calc_frame = pd.read_pickle("dataframes/r_calc_frame_notrack")
calc_frame.shape


y_train = calc_frame['TRUE_P']
x_train = calc_frame.drop(['TRUE_P','eminus_P'], axis=1)
y_train.shape

# this should display BremAdders momentum resolution (provided all dataframes match up)
plt.hist(df['eminus_P']-y_train, bins=100, range=(-5e3,5e3))

#%%
######################### TRAIN XGBOOST #####################################

# xgb_shrew = xgb.XGBRegressor(objective='reg:squarederror', max_depth=8, learning_rate=0.3, batch_size=32, verbosity=2, n_estimators=500, min_child_weight=10,
#                          reg_alpha=0.2, reg_lambda=0.3, subsample=0.8, gamma=10000)
xgb_shrew = xgb.XGBRegressor(objective='reg:squarederror', max_depth=5, learning_rate=0.3, batch_size=32, verbosity=2, n_estimators=5000,
                             reg_alpha=0.2, reg_lambda=0.65, subsample=0.8, gamma=50000)
# xgb_shrew = xgb.XGBRegressor(objective='reg:linear', max_depth=9, learning_rate=0.3, batch_size=32, verbosity=2, n_estimators=1000, min_child_weight=100,
#                          reg_alpha=0.2, reg_lambda=0.3, subsample=0.8, gamma=500000, alpha=0.2)
# regressor = xgb.XGBRegressor(learning_rate=0.3, colsample_bytree = 0.4,
#                           subsample = 0.8, objective='reg:squarederror', n_estimators=500,
#                           reg_alpha = 0.3, max_depth=9, early_stopping_rounds = 30,
#                           verbosity=2, alpha=0, gamma=1, nrounds=500)
# model = xgb.XGBRegressor(**reg_cv.best_params_)
xgb_shrew.fit(np.array(x_train),y_train)
xgb_shrew.save_model("xgb_nolabel_Shrew.model")
xgb_preds = xgb_shrew.predict(np.array(x_train))
plt.hist(calc_frame['TRUE_P']-xgb_preds, range = (-30000,30000), bins=500, alpha=0.5, color='orange')
plt.hist(calc_frame['TRUE_P']-calc_frame['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5, color='blue')
plt.show()
np.std(calc_frame['TRUE_P']-xgb_preds)
np.std(calc_frame['TRUE_P']-calc_frame['eminus_P'])

xgb.plot_importance(regressor)

########################## GPR TRAINING ##################################

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, RationalQuadratic, ConstantKernel
# kernel = DotProduct()#1**2*RBF(length_scale_bounds = (1e-1, 1e9))
# kernel = 1**2*Matern(10, (1e-5, 1e5))*ConstantKernel()
gpr_y_train = calc_frame["TRUE_P"]-calc_frame['eminus_nobrem_P']
kernel = 1**2*RBF(1e5, (1e5, 1e9))+ConstantKernel(1e6, (1e5, 1e7))
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=2000, optimizer='fmin_l_bfgs_b',n_restarts_optimizer=3)
gpr =gpr.fit(X=np.array(x_train)[:1000], y=gpr_y_train[:1000])
gpr_preds = gpr.predict(np.array(x_train))
filename = 'gpr_ShreW'
pickle.dump(gpr, open(filename, 'wb'))
gpr.get_params()
gpr_preds

range_value = 3e4
plt.hist(calc_frame['TRUE_P']-calc_frame['eminus_nobrem_P']-gpr_preds, range = (-range_value,range_value), bins=500, alpha=0.5, color='orange')
plt.hist(calc_frame['TRUE_P']-calc_frame['eminus_P'], range = (-range_value,range_value), bins=500, alpha=0.5, color='blue')
plt.show()

###################### NEURAL NET ####################################
from sklearn.neural_network import MLPRegressor

nn = MLPRegressor(hidden_layer_sizes=(256,64,64,64,32), max_iter=1500, verbose=True, validation_fraction=0.1, n_iter_no_change=40,
                  activation='relu', learning_rate_init=0.01, batch_size=128, solver='adam', learning_rate='constant', alpha=1e-5)
nn.fit(X=np.array(x_train),y=np.array(y_train))
nn.get_params
nn_preds_train = np.array(nn.predict(np.array(x_train)))
pickle.dump(nn, open("nn2_ShreW", 'wb'))
nn = pickle.load(open("nn2_ShreW", 'rb'))
plt.hist(calc_frame['TRUE_P']-nn_preds_train, range = (-30000,30000), bins=500, alpha=0.5, color='orange')
plt.hist(calc_frame['TRUE_P']-calc_frame['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5, color='blue')
plt.show()




############################## SVM ########################################
from sklearn.svm import SVR

scale_x = StandardScaler()
# scaled_x_train = x_train
scaled_x_train = scale_x.fit(x_train[:1000])
scaled_x_train = scale_x.transform(x_train)
svm_y_train = calc_frame['TRUE_P']-calc_frame['eminus_nobrem_P']
svr_rbf = SVR(kernel='rbf', C=1e3, gamma='auto', epsilon=4000, verbose=True)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
# svr_poly = SVR(kernel='poly', C=1e1, gamma='auto', degree=5, epsilon=300, verbose=True)
# svr_poly = SVR()

svr_rbf.fit(np.array(scaled_x_train)[:1000], svm_y_train[:1000])

svm_shrew_train = svr_rbf.predict(scaled_x_train)
plt.hist(calc_frame['TRUE_P']-calc_frame['eminus_nobrem_P']-svm_shrew_train+3000, range = (-30000,30000), bins=500, alpha=0.5, color='orange')
plt.hist(calc_frame['TRUE_P']-calc_frame['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5, color='blue')
plt.show()
filename = 'svr_ShreW'
pickle.dump(svr_rbf, open(filename, 'wb'))

svm_shrew_train


#%%
"""-----------------------------------------------------THIS PART IS FOR VALIDATION-------------------------------------------------------------------"""
from dataloader import Data_loader
from mathematics import Mathematics
import pandas as pd
import numpy as np
from data_handler import Data_handler
import matplotlib.pyplot as plt
import xgboost as xgb

################## LOAD VALIDATION SET ####################################
electron_usable_test = pd.read_pickle("dataframes/test_electron_usable")
class_frame_test = pd.read_pickle("dataframes/test_class_frame")


ElePhanT = xgb.XGBClassifier({'nthread':4}) #init model
ElePhanT.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ElePhant_Classifier_MT.model")

filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set_down"
df_test = Data_loader.load(filename, 100000, 110000)


handler_test = Data_handler(df_test)
predicted_frame_test = handler_test.assign_prediction(electron_usable_test, ElePhanT, class_frame_test)
calc_frame_test = handler_test.calc_frame(predicted_frame_test, n_cand=3)
calc_frame_test = pd.read_pickle("dataframes/test_calc_frame_notrack")
# calc_frame_test.to_pickle("dataframes/test_calc_frame")
calc_frame_test.shape

y_test = calc_frame_test['TRUE_P']
x_test = calc_frame_test.drop(['TRUE_P','eminus_P'], axis=1)


plt.hist(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5, color='blue')
ba_mse_test = np.average((calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'])**2)
ba_std_test = np.std(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'])
ba_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P']))
title = "MSE:", ba_mse_test, "STD:", ba_std_test
plt.title(title)
plt.show()


#%%
######################### XGBoost #####################################
xgb_shrew = xgb.XGBRegressor({'nthread':4}) #init model
xgb_shrew.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ShreW_track.model")

# x_test['label']
xgb_preds_test = xgb_shrew.predict(np.array(x_test))
range_value = 3e4
plt.hist(calc_frame_test['TRUE_P']-xgb_preds_test, range = (-range_value,range_value), bins=100, alpha=0.5, color='orange')
plt.hist(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'], range = (-range_value,range_value), bins=100, alpha=0.5, color='blue')
xgb_mse_test = np.average((calc_frame_test['TRUE_P']-xgb_preds_test)**2)
xgb_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-xgb_preds_test))
xgb_std_test = np.std(calc_frame_test['TRUE_P']-xgb_preds_test)
title = "MSE:", xgb_mse_test, "STD:", xgb_std_test
plt.title(title)
plt.show()
print(xgb_std_test/ba_std_test)
print(xgb_ae_test)
print(ba_ae_test)

#%%
########################## GPR ######################################

gpr_y_test = calc_frame_test["TRUE_P"]-calc_frame_test["eminus_nobrem_P"]
gpr_preds_test = gpr.predict(np.array(x_test))
plt.hist(calc_frame_test['TRUE_P']-calc_frame_test["eminus_nobrem_P"]-gpr_preds_test, range = (-30000,30000), bins=500, alpha=0.5, color='orange')
plt.hist(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5, color='blue')
gpr_mse_test = np.average((calc_frame_test['TRUE_P']-(gpr_preds_test+calc_frame_test['eminus_nobrem_P']))**2)
gpr_std_test = np.std(calc_frame_test['TRUE_P']-(gpr_preds_test+calc_frame_test['eminus_nobrem_P']))
title = "MSE:", gpr_mse_test, "STD:", gpr_std_test
plt.title(title)
plt.show()


########################### Neural Net #################################
nn_preds_test = np.array(nn.predict(np.array(x_test)))
plt.hist(calc_frame_test['TRUE_P']-nn_preds_test, range = (-30000,30000), bins=500, alpha=0.5, color='orange')
plt.hist(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5, color='blue')
nn_mse_test = np.average((calc_frame_test['TRUE_P']-nn_preds_test)**2)
nn_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-nn_preds_test))
nn_std_test = np.std(calc_frame_test['TRUE_P']-nn_preds_test)
title = "MSE:", nn_mse_test, "STD:", nn_std_test
plt.title(title)
plt.show()

nn_mse_test/ba_mse_test
nn_std_test/ba_std_test
nn_ae_test


####################### SVM ############################
svr_preds_test = svr_rbf.predict(scale_x.transform(x_test))
plt.hist(calc_frame_test['TRUE_P']-calc_frame_test['eminus_nobrem_P']-svr_preds_test, range = (-30000,30000), bins=500, alpha=0.5, color='orange')
plt.hist(calc_frame_test['TRUE_P']-calc_frame_test['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5, color='blue')
svr_mse_test = np.average((calc_frame_test['TRUE_P']-calc_frame_test['eminus_nobrem_P']-svr_preds_test)**2)
svr_std_test = np.std(calc_frame_test['TRUE_P']-calc_frame_test['eminus_nobrem_P']-svr_preds_test)
title = "MSE:", svr_mse_test, "STD:", svr_std_test
plt.title(title)
plt.show()


################### ENSEMBLE #########################


ensemble_preds_test = (nn.predict(np.array(x_test))+xgb_shrew.predict(np.array(x_test)))/2
plt.hist(calc_frame_test['TRUE_P']-ensemble_preds_test, range = (-30000,30000), bins=500, alpha=0.5, color='orange')
plt.hist(calc_frame_test['TRUE_P']-df_test['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5, color='blue')
ensemble_mse_test = np.average((calc_frame_test['TRUE_P']-ensemble_preds_test)**2)
ensemble_std_test = np.std(calc_frame_test['TRUE_P']-ensemble_preds_test)
title = "MSE:", ensemble_mse_test, "STD:", ensemble_std_test
plt.title(title)
plt.show()
