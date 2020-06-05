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
calc_frame_test.to_pickle("dataframes/calc_frame_test")

y_test = calc_frame_test['TRUE_P']
x_test = calc_frame_test.drop(['TRUE_P','eminus_P'], axis=1)


plt.hist(calc_frame_test['TRUE_P']-df_test['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5, color='blue')
ba_mse_test = np.average((calc_frame_test['TRUE_P']-df_test['eminus_P'])**2)
ba_std_test = np.std(calc_frame_test['TRUE_P']-df_test['eminus_P'])
ba_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-df_test['eminus_P']))
title = "MSE:", ba_mse_test, "STD:", ba_std_test
plt.title(title)
plt.show()


#%%
######################### XGBoost #####################################
xgb_shrew = xgb.XGBRegressor({'nthread':4}) #init model
xgb_shrew.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ShreW.model")


xgb_preds_test = xgb_shrew.predict(np.array(x_test))
range_value = 3e4
plt.hist(calc_frame_test['TRUE_P']-xgb_preds_test, range = (-range_value,range_value), bins=500, alpha=0.5, color='orange')
plt.hist(calc_frame_test['TRUE_P']-df_test['eminus_P'], range = (-range_value,range_value), bins=500, alpha=0.5, color='blue')
xgb_mse_test = np.average((calc_frame_test['TRUE_P']-xgb_preds_test)**2)
xgb_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-xgb_preds_test))
xgb_std_test = np.std(calc_frame_test['TRUE_P']-xgb_preds_test)
title = "MSE:", xgb_mse_test, "STD:", xgb_std_test
plt.title(title)
plt.show()
print(xgb_std_test/ba_std_test)
print(xgb_ae_test)
print(ba_ae_test)


plt.hist(calc_frame_test['TRUE_P']-xgb_preds_test, alpha=0.5, color='orange')
plt.hist(calc_frame_test['TRUE_P']-df_test['eminus_P'], alpha=0.5, color='blue')
xgb_mse_test = np.average((calc_frame_test['TRUE_P']-xgb_preds_test)**2)
xgb_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-xgb_preds_test))
xgb_std_test = np.std(calc_frame_test['TRUE_P']-xgb_preds_test)
title = "MSE:", xgb_mse_test, "STD:", xgb_std_test
plt.title(title)
plt.show()


plt.hist(calc_frame_test['TRUE_P']-xgb_preds_test, bins=500, alpha=0.5, color='orange')
plt.hist(calc_frame_test['TRUE_P']-df_test['eminus_P'], bins=500, alpha=0.5, color='blue')
xgb_mse_test = np.average((calc_frame_test['TRUE_P']-xgb_preds_test)**2)
xgb_ae_test = np.average(np.abs(calc_frame_test['TRUE_P']-xgb_preds_test))
xgb_std_test = np.std(calc_frame_test['TRUE_P']-xgb_preds_test)
title = "MSE:", xgb_mse_test, "STD:", xgb_std_test
plt.title(title)
plt.show()









#####################################
