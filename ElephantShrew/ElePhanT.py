from dataloader import Data_loader
from mathematics import Mathematics
import pandas as pd
import numpy as np
from data_handler import Data_handler
import xgboost as xgb
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix

#%%
########################## TRAINING #################################
#
class_frame = pd.read_pickle("dataframes/c_class_frame_down")

# Create Labels/Training DataFrame

y_train = class_frame['label']*1
x_train = class_frame.drop(['label'], axis=1)*1
x_train.shape
# train classifier

scaling = (len(y_train)-np.sum(y_train))/np.sum(y_train)
# scaling = 1
# model = xgb.XGBClassifier(verbosity=2)

model = xgb.XGBClassifier(learning_rate=0.3, colsample_bytree = 0.4,
                          subsample = 0.8, objective='binary:logistic', n_estimators=5000,
                          reg_alpha = 0.3, max_depth=5, early_stopping_rounds = 10,
                          verbosity=2, scale_pos_weight=0.65, alpha=0.3, gamma=5, nrounds=1000)
model.fit(np.array(x_train),np.array(y_train).astype(int))
np.sum(class_frame['label'])
np.sum(test_class_frame['label'])
# get statistics
xgb.plot_importance(model)
model.predict(np.array(x_train))
tn, fp, fn, tp = confusion_matrix(np.array(y_train).astype(int), np.argmax(model.predict_proba(np.array(x_train)), axis=1)).ravel()
print("True Positives", tp, "\nTrue Negatives:", tn,  "\nFalse Positives:", fp, "\nFalse Negatives", fn)

print("Total Predictions:", tn+fp+fn+tp)
model.save_model('ElePhant_Classifier_MT.model')





# #%%
#
#
#
# ################## Verification ###############################
#

filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set_down"
test_df = Data_loader.load(filename, 100000, 110000)
test_handler = Data_handler(test_df)

test_electron_usable = pd.read_pickle("dataframes/test_electron_usable")
test_class_frame = pd.read_pickle("dataframes/test_class_frame")

model = xgb.XGBClassifier({'nthread':4}) #init model
# model.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ElePhant_Classifier_Competitive.model")

model.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ElePhant_Classifier.model")

predicted_frame_test = test_handler.assign_prediction(test_electron_usable, model, test_class_frame)

xgb.plot_importance(model)

y_test = test_class_frame['label']*1
x_test = test_class_frame.drop(['label'], axis=1)*1

tn, fp, fn, tp = confusion_matrix(np.array(y_test).astype(int), np.argmax(model.predict_proba(np.array(x_test)), axis=1)).ravel()
print("True Positives", tp, "\nTrue Negatives:", tn,  "\nFalse Positives:", fp, "\nFalse Negatives", fn)

tn, fp, fn, tp = confusion_matrix(np.array(y_test).astype(int), np.argmax(model.predict_proba(np.array(x_test)), axis=1)).ravel()
print("True Positives", tp, "\nTrue Negatives:", tn,  "\nFalse Positives:", fp, "\nFalse Negatives", fn)

test_df.columns.tolist()



#%%
############################ PERFORMANCE TESTING ##############################


"""Finds relativistic momentum"""
def rel_mom(energy):
    c = 3e8
    return np.array(energy)/c



brem_tp = []
brem_fp = []
brem_tn = []
brem_fn = []
event_brem_energy = []
elephant_tp = []
elephant_fp = []
elephant_tn = []
elephant_fn = []
event_elephant_energy = []
event_brem_p = []
event_elephant_p = []

for i in range(10000):
    photon_energy = []
    event_brem_energy.append([])
    event_elephant_energy.append([])
    event_brem_p.append([])
    event_elephant_p.append([])
    for j in range(len(predicted_frame_test['ecal_clusters'][i])):
        if predicted_frame_test['ecal_clusters'][i][j]["xgb_pred"]>0.5:
            event_elephant_energy[i].append(predicted_frame_test['ecal_clusters'][i][j]["energy"])
            if predicted_frame_test['ecal_clusters'][i][j]["label"]==True:
                elephant_tp.append(predicted_frame_test['ecal_clusters'][i][j]["energy"])
            else:
                elephant_fp.append(predicted_frame_test['ecal_clusters'][i][j]["energy"])
        else:
            if predicted_frame_test['ecal_clusters'][i][j]["label"]==False:
                elephant_tn.append(predicted_frame_test['ecal_clusters'][i][j]["energy"])
            else:
                elephant_fn.append(predicted_frame_test['ecal_clusters'][i][j]["energy"])
        if predicted_frame_test['ecal_clusters'][i][j]["label"]==True:
            photon_energy.append(predicted_frame_test['ecal_clusters'][i][j]["energy"])


    # if brem found photons:
    if int(test_df['eminus_BremMultiplicity'][i]) > 0:
        # if there are photons and bremadder predicted the correct amount
        if len(photon_energy)==int(test_df['eminus_BremMultiplicity'][i]):
            for j in range(len(photon_energy)):
                brem_tp.append(photon_energy[j])
                event_brem_energy[i].append(predicted_frame_test['ecal_clusters'][i][j]["energy"])

            for j in range(len(predicted_frame_test['ecal_clusters'][i])):
                if predicted_frame_test['ecal_clusters'][i][j]["label"]==False:
                    brem_tn.append(predicted_frame_test['ecal_clusters'][i][j]["energy"])


        # if there are less photons than predicted
        if len(photon_energy)<int(test_df['eminus_BremMultiplicity'][i]):
            for j in range(len(photon_energy)):
                brem_tp.append(predicted_frame_test['ecal_clusters'][i][j]["energy"])

            for j in range(len(photon_energy), int(test_df['eminus_BremMultiplicity'][i])):
                brem_fp.append(predicted_frame_test['ecal_clusters'][i][j]["energy"])
                event_brem_energy[i].append(predicted_frame_test['ecal_clusters'][i][j]["energy"])

            for j in range(int(test_df['eminus_BremMultiplicity'][i]), len(predicted_frame_test['ecal_clusters'][i])):
                brem_tn.append(predicted_frame_test['ecal_clusters'][i][j]["energy"])


        # if there are more photons than brem predicted
        if len(photon_energy)>int(test_df['eminus_BremMultiplicity'][i]):
            # append as many as brem predicted to true positive
            for j in range(int(test_df['eminus_BremMultiplicity'][i])):
                brem_tp.append(photon_energy[j])
                event_brem_energy[i].append(photon_energy[j])

            # append the ones it didn't predict as false negatives
            for j in range(int(test_df['eminus_BremMultiplicity'][i]), len(photon_energy)):
                brem_fn.append(photon_energy[j])

            # append rest as true negatives
            for j in range(len(predicted_frame_test['ecal_clusters'][i])):
                if predicted_frame_test['ecal_clusters'][i][j]["label"]==False:
                    brem_tn.append(predicted_frame_test['ecal_clusters'][i][j]["energy"])
    else:
        # if brem predicted no photon and there is none
        if len(photon_energy)==0:
            for j in range(len(predicted_frame_test['ecal_clusters'][i])):
                brem_tn.append(predicted_frame_test['ecal_clusters'][i][j]["energy"])
        # brem predicted no photon but there is one
        else:
            for j in range(len(photon_energy)):
                brem_fn.append(photon_energy[j])

            # append rest as true negatives
            for j in range(len(predicted_frame_test['ecal_clusters'][i])):
                if predicted_frame_test['ecal_clusters'][i][j]["label"]==False:
                    brem_tn.append(predicted_frame_test['ecal_clusters'][i][j]["energy"])
    event_brem_energy[i] = np.sum(event_brem_energy[i])
    event_elephant_energy[i] = np.sum(event_elephant_energy[i])
    event_elephant_p[i] = np.sum(rel_mom(event_elephant_energy[i]))
    event_brem_p[i] = np.sum(rel_mom(event_brem_energy[i]))
    # shrew_error += abs(true_temp-shrew_temp)
    # brem_error += abs(true_temp-int(test_df['eminus_BremMultiplicity'][i]))

len(brem_tp)
len(brem_fp)
len(brem_tn)
len(brem_fn)
len(brem_tp)+len(brem_fp)+len(brem_tn) +len(brem_fn)

len(elephant_tp)
len(elephant_fp)
len(elephant_tn)
len(elephant_fn)

len(elephant_tp)+len(elephant_fp)+len(elephant_tn)+len(elephant_fn)


def run_on_demand():
    brem_precision = len(brem_tp)/(len(brem_tp)+len(brem_fp))
    brem_precision

    elephant_precision = len(elephant_tp)/(len(elephant_tp)+len(elephant_fp))
    elephant_precision

    brem_recall = len(brem_tp)/(len(brem_tp)+len(brem_fn))
    brem_recall

    elephant_recall = len(elephant_tp)/(len(elephant_tp)+len(elephant_fn))
    elephant_recall

    brem_f1 = 2*brem_recall*brem_precision/(brem_recall+brem_precision)
    brem_f1

    elephant_f1 = 2*elephant_precision*elephant_recall/(elephant_precision+elephant_recall)
    elephant_f1

def run_on_demand():
    brem_precision = np.sum(brem_tp)/(np.sum(brem_tp)+np.sum(brem_fp))
    brem_precision

    elephant_precision = np.sum(elephant_tp)/(np.sum(elephant_tp)+np.sum(elephant_fp))
    elephant_precision

    brem_recall = np.sum(brem_tp)/(np.sum(brem_tp)+np.sum(brem_fn))
    brem_recall

    elephant_recall = np.sum(elephant_tp)/(np.sum(elephant_tp)+np.sum(elephant_fn))
    elephant_recall

    brem_f1 = 2*brem_recall*brem_precision/(brem_recall+brem_precision)
    brem_f1

    elephant_f1 = 2*elephant_precision*elephant_recall/(elephant_precision+elephant_recall)
    elephant_f1

def run_on_demand():
    1-(np.sum(elephant_fp)+np.sum(elephant_fn))/(np.sum(brem_fp)+np.sum(brem_fn))
    1-(len(elephant_fp)+len(elephant_fn))/(len(brem_fp)+len(brem_fn))



incorrect_elephant = np.concatenate((elephant_fp, elephant_fn))
incorrect_brem = np.concatenate((brem_fp, brem_fn))

fig, ax = plt.subplots()
brem_tp_hist = ax.hist(brem_tp, alpha=0.5, range=(0,1.5e4), color='blue', bins=100, label='BremAdder TP')
ele_tp_hist = ax.hist(elephant_tp, alpha=0.5, range=(0,1.5e4), color='orange', bins=100, label='ElePhanT TP')
np.sum(elephant_tp)/np.sum(brem_tp)
ax.legend()
plt.xlabel("Energy")
title = "Histogram of TP Energy | Change: " + str(np.round((np.sum(elephant_tp)/np.sum(brem_tp)-1)*100, decimals=1)) + "%"
plt.title(title)
plt.show()

fig, ax = plt.subplots()
brem_fn_hist = ax.hist(brem_fn, alpha=0.5, range=(0,1.5e4), color='blue', bins=100, label='BremAdder FN')
ele_fn_hist = ax.hist(elephant_fn, alpha=0.5, range=(0,1.5e4), color='orange', bins=100, label='ElePhanT FN')
np.sum(elephant_fn)/np.sum(brem_fn)
ax.legend()
plt.xlabel("Energy")
title = "Histogram of FN Energy | Change: " + str(np.round((np.sum(elephant_fn)/np.sum(brem_fn)-1)*100, decimals=1)) + "%"
plt.title(title)
plt.show()

fig, ax = plt.subplots()
brem_fp_hist = ax.hist(brem_fp, alpha=0.5, range=(0,1.5e4), color='blue', bins=100, label='BremAdder FP')
ele_fp_hist = ax.hist(elephant_fp, alpha=0.5, range=(0,1.5e4), color='orange', bins=100, label='ElePhanT FP')
np.sum(elephant_fp)/np.sum(brem_fp)
ax.legend()
plt.xlabel("Energy")
title = "Histogram of FP Energy | Change: " + str(np.round((np.sum(elephant_fp)/np.sum(brem_fp)-1)*100, decimals=1)) + "%"
plt.title(title)
plt.show()


fig, ax = plt.subplots()
brem_tn_hist = ax.hist(brem_tn, alpha=0.5, range=(0,1.5e4), color='blue', bins=100, label='BremAdder TN')
ele_tn_hist = ax.hist(elephant_tn, alpha=0.5, range=(0,1.5e4), color='orange', bins=100, label='ElePhanT TN')
np.sum(elephant_tn)/np.sum(brem_tn)
ax.legend()
plt.xlabel("Energy")
title = "Histogram of TN Energy | Change: " + str(np.round((np.sum(elephant_tn)/np.sum(brem_tn)-1)*100, decimals=1)) + "%"
plt.title(title)
# plt.text(s="Total Brem Energy:", x=10000, y=10000)
plt.show()



fig, ax = plt.subplots()
### let's make a histogram of the differences in energy
brem_errors = np.concatenate((brem_fp, np.multiply(-1,brem_fn),np.multiply(0,brem_tp)))


ax.hist(brem_errors, alpha=0.5, range=(-5e3,5e3), color='blue', bins=50, label='BremAdder')
np.multiply(np.random.rand(len(elephant_tp))*0.2-0.1,elephant_tp)

elephant_errors = np.concatenate((elephant_fp, np.multiply(-1,elephant_fn),np.multiply(0,elephant_tp)))

ax.hist(elephant_errors, alpha=0.5, range=(-5e3,5e3), color='orange', bins=50, label='ElePhanT')
ax.legend()
plt.xlabel("Approximated Momentum Resolution")
plt.title("Momentum Resolution Improvement")
plt.show()

true_P = np.sqrt(test_df['eminus_TRUEP_X']**2+test_df['eminus_TRUEP_Y']**2+test_df['eminus_TRUEP_Z']**2)
brem_resolution = test_df['eminus_P']-true_P


fig, ax = plt.subplots()
ax.hist(brem_resolution, alpha=0.5, color='green', range=(-5e3,5e3), bins=50, label='True BremAdder')
ax.hist(brem_errors, alpha=0.5, range=(-5e3,5e3), color='blue', bins=50, label='Approximated BremAdder')
plt.xlabel("Momentum Resolution")
plt.title("True vs Approximated Momentum Resolution")
plt.legend()
plt.show()



res_hist = ax.hist(brem_resolution, alpha=0.5, color='green', range=(-5e3,5e3), bins=50, label='True BremAdder')
brem_error_hist = ax.hist(brem_errors, alpha=0.5, range=(-5e3,5e3), color='blue', bins=50, label='Approximated BremAdder')
1-np.sum(np.abs((res_hist[0]-brem_error_hist[0])))/np.sum(res_hist[0])


### let's make a histogram of the differences in energy
# brem_errors = np.concatenate((brem_fp, np.multiply(-1,brem_fn),np.multiply(0,brem_tp)))
#
noise = 0.2
bias = -0.05
background = 0.07
# np.multiply(1+ (np.random.normal(size=len(brem_fp))*noise+bias),brem_fp)

fig, ax = plt.subplots()
random_error = np.random.rand(int((len(brem_fp)+len(brem_fn)+len(brem_tp))*background))*(2*5e3)-5e3

brem_errors = np.concatenate((np.multiply(1+ (np.random.normal(size=len(brem_fp))*noise+bias),brem_fp),
                                  np.multiply(-(1+(np.random.normal(size=len(brem_fn))*noise+bias)),brem_fn),
                                  np.multiply(np.random.normal(size=len(brem_tp))*noise+bias,brem_tp),
                                  random_error))
ax.hist(brem_errors, alpha=0.5, range=(-5e3,5e3), color='blue', bins=50, label='BremAdder')

# elephant_errors = np.concatenate((elephant_fp, np.multiply(-1,elephant_fn),np.multiply(0,elephant_tp)))
elephant_errors = np.concatenate((np.multiply(1+ (np.random.normal(size=len(elephant_fp))*noise+bias),elephant_fp),
                                  np.multiply(-(1+(np.random.normal(size=len(elephant_fn))*noise+bias)),elephant_fn),
                                  np.multiply(np.random.normal(size=len(elephant_tp))*noise+bias,elephant_tp),
                                  random_error))
ax.hist(elephant_errors, alpha=0.5, range=(-5e3,5e3), color='orange', bins=50, label='ElePhanT')

ax.legend()
plt.xlabel("Approximated Momentum Resolution")
plt.title("Momentum Resolution Improvement")
plt.show()


brem_avg = np.average(np.abs(brem_errors))
elephant_avg = np.average(np.abs(elephant_errors))
brem_std = np.std(brem_errors)
elephant_std = np.std(elephant_errors)
elephant_std/brem_std
elephant_avg**2/brem_avg**2
brem_std
# np.std(brem_resolution)
fig, ax = plt.subplots()
ax.hist(brem_resolution, alpha=0.5, color='green', range=(-5e3,5e3), bins=50, label='True BremAdder')
ax.hist(brem_errors, alpha=0.5, range=(-5e3,5e3), color='blue', bins=50, label='Approximated BremAdder')
plt.xlabel("Momentum Resolution")
plt.title("True vs Approximated Momentum Resolution")
plt.legend()
plt.show()






#################### TRY TO FIND ACTUAL MOMENTUM RESOLUTION BASED ON RELATIVISTIC Momentum

plt.hist(brem_resolution, alpha=0.5, color='green', range=(-1e5,1e5), bins=50, label='True BremAdder')

resolution = true_P-test_df['eminus_nobrem_P']
# plt.hist(resolution-np.sqrt(np.array(event_brem_energy))*170, alpha=0.5, color='blue', range=(-1e5,1e5), bins=50)
plt.hist(resolution-np.sqrt(np.array(event_elephant_energy))*170, alpha=0.5, color='orange', range=(-1e5,1e5), bins=50)
brem_std_2 = np.std(resolution-np.sqrt(np.array(event_brem_energy))*170)
elephant_std_2 = np.std(resolution-np.sqrt(np.array(event_elephant_energy))*170)
brem_std_2
elephant_std_2


# plt.hist(resolution-np.sqrt(np.array(event_brem_p)), alpha=0.5, color='blue', range=(-1e5,1e5), bins=50)
plt.hist(resolution, alpha=0.5, color='green', range=(-1e5,1e5), bins=100)
plt.hist(resolution-np.array(event_elephant_p)*1e9, range=(-1e5,1e5), alpha=0.5, color='orange', bins=100)
np.std(resolution-np.array(event_elephant_p)*1e9)


plt.hist(resolution-np.array(event_elephant_energy)*0, alpha=0.5, range=(-5e3,5e3), color='orange', bins=50)




np.array(resolution), np.array(event_elephant_energy)




event_elephant_energy
















tot_hist = plt.hist(total_clusters, alpha=0.5, range=(0,1.5e4), color='green', bins=100)
ele_hist = plt.hist(correct_elephant, alpha=0.5, range=(0,1.5e4), color='orange', bins=100)
brem_hist = plt.hist(correct_brem, alpha=0.5, range=(0,1.5e4), bins=100)
plt.yscale('log')


plt.hist(incorrect_elephant, alpha=0.5, range=(0,1.5e4), color='orange', bins=100)
plt.hist(incorrect_brem, alpha=0.5, range=(0,1.5e4), bins=100)

~np.sum(correct_elephant)
np.sum(correct_brem)
np.sum(correct_elephant)/np.sum(correct_brem)
np.sum(incorrect_elephant)
np.sum(incorrect_brem)
np.sum(incorrect_elephant)/np.sum(incorrect_brem)


ele_perc = ele_hist[0]/tot_hist[0]
plt.bar(np.arange(100), ele_perc)

brem_perc = brem_hist[0]/tot_hist[0]
plt.bar(np.arange(100), brem_perc)

plt.bar(np.arange(100), ele_perc-brem_perc)


plt.bar(np.arange(100),ele_hist[0]-brem_hist[0])
# plt.yscale('log')
