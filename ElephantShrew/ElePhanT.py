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
########################## TRAINING #################################

# Create all dataframes
filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set"
df = Data_loader.load(filename, 0, 10000)
training_handler = Data_handler(df)
usable = training_handler.return_usable(df)
usable.to_pickle("dataframes/usable")
filtered_usable = training_handler.filter_clusters(usable)
filtered_usable.to_pickle("dataframes/filtered_usable")
dict_usable = training_handler.cluster_to_dict(filtered_usable, matching_region=100)
dict_usable.to_pickle("dataframes/dict_usable")
electron_usable = training_handler.is_electron(dict_usable)
electron_usable = training_handler.remove_electron_label(electron_usable)
electron_usable.to_pickle("dataframes/electron_usable")
class_frame = training_handler.prepare_classification(electron_usable)
class_frame.to_pickle("dataframes/class_frame")

class_frame = pd.read_pickle("dataframes/class_frame")

# Create Labels/Training DataFrame

y_train = class_frame['label']*1
x_train = class_frame.drop(['label'], axis=1)*1
x_train.shape
# train classifier
model = xgb.XGBClassifier(scale_pos_weight=1, learning_rate=0.1, colsample_bytree = 0.4,
                          subsample = 0.8, objective='binary:logistic', n_estimators=1000,
                          reg_alpha = 0.3, max_depth=6, gamma=1, early_stopping_rounds = 1000)
model.fit(np.array(x_train),np.array(y_train).astype(int))


# get statistics
xgb.plot_importance(model)
model.predict(np.array(x_train))
tn, fp, fn, tp = confusion_matrix(np.array(y_train).astype(int), np.argmax(model.predict_proba(np.array(x_train)), axis=1)).ravel()
print("True Positives", tp, "\nTrue Negatives:", tn,  "\nFalse Positives:", fp, "\nFalse Negatives", fn)
print("Total Predictions:", tn+fp+fn+tp)
model.save_model('ElePhant_Classifier.model')








#%%


################## Verification ###############################


filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set"
test_df = Data_loader.load(filename, 10000, 20000)
test_handler = Data_handler(test_df)
test_usable = test_handler.return_usable(test_df)
test_usable.to_pickle("dataframes/test_usable")
test_filtered_usable = test_handler.filter_clusters(test_usable)
test_filtered_usable.to_pickle("dataframes/test_filtered_usable")
test_dict_usable = test_handler.cluster_to_dict(test_filtered_usable)
test_dict_usable.to_pickle("dataframes/test_dict_usable")
test_electron_usable = test_handler.is_electron(test_dict_usable)
test_electron_usable.to_pickle("dataframes/test_electron_usable")
test_electron_usable = test_handler.remove_electron_label(test_electron_usable)
test_electron_usable.to_pickle("dataframes/test_electron_usable")
test_class_frame = test_handler.prepare_classification(test_electron_usable)
test_class_frame.to_pickle("dataframes/test_class_frame")
test_class_frame


test_electron_usable = pd.read_pickle("dataframes/test_electron_usable")
test_class_frame = pd.read_pickle("dataframes/test_class_frame")

model = xgb.XGBClassifier({'nthread':4}) #init model
model.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ElePhant_Classifier.model")

test_electron_usable

predicted_frame_test = test_handler.assign_prediction(test_electron_usable, model, test_class_frame)
predicted_frame_test.columns



y_test = test_class_frame['label']*1
x_test = test_class_frame.drop(['label'], axis=1)*1

tn, fp, fn, tp = confusion_matrix(np.array(y_test).astype(int), np.argmax(model.predict_proba(np.array(x_test)), axis=1)).ravel()
print("True Positives", tp, "\nTrue Negatives:", tn,  "\nFalse Positives:", fp, "\nFalse Negatives", fn)

test_df.columns.tolist()
for i in range(20):
    # for j in range(len(test_df['ECAL_photon_x_arr'][i])):
    plt.scatter(test_df['ECAL_cluster_x_arr'][i],test_df['ECAL_cluster_y_arr'][i], c='black')
    plt.scatter(test_df['eminus_MCphotondaughters_ECAL_X'][i],test_df['eminus_MCphotondaughters_ECAL_Y'][i], c='orange')
    for j in range(len(predicted_frame_test['ecal_clusters'][i])):
        if predicted_frame_test['ecal_clusters'][i][j]["xgb_pred"]>0.5:
            plt.scatter(predicted_frame_test['ecal_clusters'][i][j]["x_pos"],predicted_frame_test['ecal_clusters'][i][j]["y_pos"], color='red')
        if predicted_frame_test['ecal_clusters'][i][j]["is_electron"]==True:
            plt.scatter(predicted_frame_test['ecal_clusters'][i][j]["x_pos"],predicted_frame_test['ecal_clusters'][i][j]["y_pos"], color='green', marker='3')
    plt.title(str(test_df['eminus_BremMultiplicity'][i]))
    plt.show()





#%%
############################ PERFORMANCE TESTING ##############################
