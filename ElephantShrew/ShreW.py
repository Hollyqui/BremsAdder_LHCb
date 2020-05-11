from dataloader import Data_loader
from mathematics import Mathematics
import pandas as pd
import numpy as np
from data_handler import Data_handler
import matplotlib.pyplot as plt
import xgboost as xgb


electron_usable = pd.read_pickle("dataframes/test_electron_usable")
class_frame = pd.read_pickle("dataframes/test_class_frame")
model = xgb.XGBClassifier({'nthread':4}) #init model
model.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ElePhant_Classifier.model")

filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set"
df = Data_loader.load(filename, 10000, 20000)


handler = Data_handler(df)
predicted_frame = handler.assign_prediction(electron_usable, model, class_frame)
calc_frame = handler.calc_frame(predicted_frame, n_cand=3)

calc_frame.columns
y_train = calc_frame['TRUE_P']-calc_frame['eminus_nobrem_P']
x_train = calc_frame.drop(['TRUE_P'], axis=1)


# regressor = xgb.XGBRegressor(objective='reg:squarederror', verbosity=2)
regressor = xgb.XGBRegressor(objective='reg:linear', max_depth=8, learning_rate=0.3, batch_size=32, verbosity=2, n_estimators=1000, min_child_weight=10,
                         reg_alpha=0.2, reg_lambda=0.3, subsample=0.8, gamma=100000000)

# model = xgb.XGBRegressor(**reg_cv.best_params_)
regressor.fit(np.array(x_train),y_train)

preds = regressor.predict(np.array(x_train))
plt.hist(y_train-preds, range = (-30000,30000), bins=500, alpha=0.5, color='orange')
plt.hist(calc_frame['TRUE_P']-df['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5, color='blue')
plt.show()

xgb.plot_importance(model)


#%%
################# VALIDATION #############################




electron_usable_test = pd.read_pickle("dataframes/electron_usable")
class_frame_test = pd.read_pickle("dataframes/class_frame")
model = xgb.XGBClassifier({'nthread':4}) #init model
model.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ElePhant_Classifier.model")

filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set"
df_test = Data_loader.load(filename, 10000, 20000)


handler_test = Data_handler(df_test)
predicted_frame_test = handler.assign_prediction(electron_usable_test, model, class_frame_test)
calc_frame_test = handler.calc_frame(predicted_frame_test, n_cand=3)

calc_frame.columns
y_test = calc_frame_test['TRUE_P']
x_test = calc_frame_test.drop(['TRUE_P'], axis=1)


preds_test = regressor.predict(np.array(x_test))
plt.hist(y_test-preds_test, range = (-30000,30000), bins=500, alpha=0.5)
plt.hist(calc_frame_test['TRUE_P']-df['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5)
plt.show()
np.average(abs(y_train-preds))
np.average(abs(y_train-df['eminus_P']))
