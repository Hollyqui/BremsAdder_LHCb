from dataloader import Data_loader
from mathematics import Mathematics
import pandas as pd
import numpy as np
from data_handler import Data_handler
import matplotlib.pyplot as plt
import xgboost as xgb



############## PREPARE DATASET ########################

electron_usable = pd.read_pickle("dataframes/electron_usable_down")
class_frame = pd.read_pickle("dataframes/class_frame_down")
model = xgb.XGBClassifier({'nthread':4}) #init model
model.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ElePhant_Classifier.model")

filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set_down"
df = Data_loader.load(filename, 0, 100000)


handler = Data_handler(df)
predicted_frame = handler.assign_prediction(electron_usable, model, class_frame)
calc_frame = handler.calc_frame(predicted_frame, n_cand=3)
calc_frame.to_pickle("dataframes/calc_frame")
calc_frame.shape
calc_frame.columns
y_train = calc_frame['TRUE_P']#-calc_frame['eminus_nobrem_P']
x_train = calc_frame.drop(['TRUE_P'], axis=1)
y_train.shape

######################### TRAIN XGBOOST #####################################

# regressor = xgb.XGBRegressor(objective='reg:squarederror', verbosity=2)
regressor = xgb.XGBRegressor(objective='reg:linear', max_depth=8, learning_rate=0.3, batch_size=32, verbosity=2, n_estimators=500, min_child_weight=10,
                         reg_alpha=0.2, reg_lambda=0.3, subsample=0.8, gamma=10000)
# regressor = xgb.XGBRegressor(learning_rate=0.3, colsample_bytree = 0.4,
#                           subsample = 0.8, objective='reg:squarederror', n_estimators=500,
#                           reg_alpha = 0.3, max_depth=9, early_stopping_rounds = 30,
#                           verbosity=2, alpha=0, gamma=1, nrounds=500)
# model = xgb.XGBRegressor(**reg_cv.best_params_)
regressor.fit(np.array(x_train),y_train)
regressor.save_model("ShreW.model")
preds = regressor.predict(np.array(x_train))
plt.hist(calc_frame['TRUE_P']-preds, range = (-30000,30000), bins=500, alpha=0.5, color='orange')
plt.hist(calc_frame['TRUE_P']-df['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5, color='blue')
plt.show()
np.std(calc_frame['TRUE_P']-preds)
np.std(calc_frame['TRUE_P']-df['eminus_P'])

xgb.plot_importance(model)


#%%
################# VALIDATION #############################



################## LOAD VALIDATION SET ####################################
electron_usable_test = pd.read_pickle("dataframes/test_electron_usable")
class_frame_test = pd.read_pickle("dataframes/test_class_frame")


model = xgb.XGBClassifier({'nthread':4}) #init model
model.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ElePhant_Classifier.model")

filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set_down"
df_test = Data_loader.load(filename, 100000, 110000)


handler_test = Data_handler(df_test)
predicted_frame_test = handler_test.assign_prediction(electron_usable_test, model, class_frame_test)
calc_frame_test = handler_test.calc_frame(predicted_frame_test, n_cand=3)
calc_frame_test.to_pickle("dataframes/calc_frame_test")

y_test = calc_frame_test['TRUE_P']
x_test = calc_frame_test.drop(['TRUE_P'], axis=1)


plt.hist(calc_frame_test['TRUE_P']-df_test['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5, color='blue')
ba_mse = np.average((calc_frame_test['TRUE_P']-df_test['eminus_P'])**2)
ba_std = np.std(calc_frame_test['TRUE_P']-df_test['eminus_P'])
title = "MSE:", ba_mse, "STD:", ba_std
plt.title(title)
plt.show()

######################### XGBoost #####################################
xgb_shrew = xgb.XGBRegressor({'nthread':4}) #init model
xgb_shrew.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ShreW.model")



xgb_preds_test = xgb_shrew.predict(np.array(x_test))
plt.hist(calc_frame_test['TRUE_P']-xgb_preds_test, range = (-30000,30000), bins=500, alpha=0.5, color='orange')
plt.hist(calc_frame_test['TRUE_P']-df_test['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5, color='blue')
xgb_mse = np.average((calc_frame_test['TRUE_P']-preds_test)**2)
xgb_std = np.std(calc_frame_test['TRUE_P']-preds_test)
title = "MSE:", xgb_mse, "STD:", xgb_std
plt.title(title)
plt.show()