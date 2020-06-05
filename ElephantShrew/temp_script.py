from dataloader import Data_loader
from mathematics import Mathematics
import pandas as pd
import numpy as np
from data_handler import Data_handler
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix

#%%
########################## ClASSIFIER TRAINING #################################

filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set_down"
c_df_down = Data_loader.load(filename, 0, 50000)
c_df_down.shape


c_training_handler = Data_handler(c_df_down)
c_usable_down = c_training_handler.return_usable(c_df_down)
c_usable_down.to_pickle("dataframes/c_usable_down")
c_filtered_usable_down = c_training_handler.filter_clusters(c_usable_down)
c_filtered_usable_down.to_pickle("dataframes/c_filtered_usable_down")
c_dict_usable_down = c_training_handler.dictonarize(c_filtered_usable_down, matching_region=100)

# dict_usable_down.reset_index().drop(['index'], axis=1)

c_dict_usable_down.to_pickle("dataframes/c_dict_usable_down")
c_electron_usable_down = c_training_handler.is_electron(c_dict_usable_down)
c_electron_usable_down = c_training_handler.remove_electron_label(c_electron_usable_down)
c_electron_usable_down.to_pickle("dataframes/c_electron_usable_down")
c_class_frame_down = c_training_handler.prepare_classification(c_electron_usable_down)
c_class_frame_down.to_pickle("dataframes/c_class_frame_down")
np.sum(c_class_frame_down['label'])


############################### REGRESSOR TRAINING #############################


filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set_down"
r_df_down = Data_loader.load(filename, 50000, 100000)
r_df_down.shape


r_training_handler = Data_handler(r_df_down)
r_usable_down = r_training_handler.return_usable(r_df_down)
r_usable_down.to_pickle("dataframes/r_usable_down")
r_filtered_usable_down = r_training_handler.filter_clusters(r_usable_down)
r_filtered_usable_down.to_pickle("dataframes/r_filtered_usable_down")
r_dict_usable_down = r_training_handler.dictonarize(r_filtered_usable_down, matching_region=100)

# dict_usable_down.reset_index().drop(['index'], axis=1)

r_dict_usable_down.to_pickle("dataframes/r_dict_usable_down")
r_electron_usable_down = r_training_handler.is_electron(r_dict_usable_down)
r_electron_usable_down = r_training_handler.remove_electron_label(r_electron_usable_down)
r_electron_usable_down.to_pickle("dataframes/r_electron_usable_down")
r_class_frame_down = r_training_handler.prepare_classification(r_electron_usable_down)
r_class_frame_down.to_pickle("dataframes/r_class_frame_down")
np.sum(class_frame_down['label'])




##################### TEST #####################################

filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set_down"
test_df = Data_loader.load(filename, 100000, 110000)
test_handler = Data_handler(test_df)
test_usable = test_handler.return_usable(test_df)
test_usable.to_pickle("dataframes/test_usable")
test_filtered_usable = test_handler.filter_clusters(test_usable)
test_filtered_usable.to_pickle("dataframes/test_filtered_usable")

test_dict_usable = test_handler.dictonarize(test_filtered_usable, matching_region=100)

test_electron_usable = pd.read_pickle("dataframes/test_electron_usable")
test_class_frame = pd.read_pickle("dataframes/test_class_frame")

test_dict_usable.to_pickle("dataframes/test_dict_usable")
test_electron_usable = test_handler.is_electron(test_dict_usable)
test_electron_usable.to_pickle("dataframes/test_electron_usable")
test_electron_usable = test_handler.remove_electron_label(test_electron_usable)
test_electron_usable.to_pickle("dataframes/test_electron_usable")
test_class_frame = test_handler.prepare_classification(test_electron_usable)
test_class_frame.to_pickle("dataframes/test_class_frame")

# now load model to classify all clusters (ElePhanT)
model = xgb.XGBClassifier({'nthread':4})
model.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ElePhant_Classifier.model")


# make predictions and create classification frame
predicted_frame_test = test_handler.assign_prediction(test_electron_usable, model, test_class_frame)
calc_frame_test = test_handler.calc_frame(predicted_frame_test, n_cand=3)
calc_frame_test.to_pickle("dataframes/test_calc_frame_notrack")
calc_frame_test.shape
print(calc_frame_test['scaled_energy'],calc_frame_test['energy'])
# double check that everything is correct (then the BremAdder Momentum resolution should be shown now)
plt.hist(calc_frame_test['TRUE_P']-test_df['eminus_P'], range = (-30000,30000), bins=500, alpha=0.5, color='blue')
