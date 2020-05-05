try:
    from Code.xgboosting import *
    from Code.refactored_preprocessing import *
except:
    from xgboosting import *
    from refactored_preprocessing import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#### load dataframe ######

print("imports successful")

filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set"
df = pd.read_pickle(filename)


START_ROW = 0
MAX_ROWS = 500
df_orig = pd.read_pickle(filename)
df_orig = df[START_ROW:MAX_ROWS+START_ROW]
len(df_orig)
df_orig = df.reset_index()

df = df[START_ROW:MAX_ROWS+START_ROW]
len(df)
df = df.reset_index()


##### candidate frame creation stats #############

DAUGHTER_CLUSTER_MATCH_MAX=75
n_candidate = 10
    # returns a dataframe where each row corresponds to one candidate cluster

training_frame = False # specifies whether frame can be used for training
df_backup = copy.deepcopy(df)
ex = df['eminus_ECAL_x']
ey = df['eminus_ECAL_y']
clusters_x = df['ECAL_cluster_x_arr']
clusters_y = df['ECAL_cluster_y_arr']

daughters_x = df['eminus_MCphotondaughters_ECAL_X']
daughters_y = df['eminus_MCphotondaughters_ECAL_Y']
training_frame = True

photon_clusters_x = df['ECAL_photon_x_arr']
photon_clusters_y = df['ECAL_photon_y_arr']
velo_x = df['eminus_ECAL_velotrack_x']
velo_y = df['eminus_ECAL_velotrack_y']
ttrack_x = df['eminus_ECAL_TTtrack_x']
ttrack_y = df['eminus_ECAL_TTtrack_y']
line_x, line_y = (velo_x+ttrack_x)/2, (velo_y+ttrack_y)/2

ttrack_state_x = df['eminus_TTstate_x']
ttrack_state_y = df['eminus_TTstate_y']
ttrack_state_z = df['eminus_TTstate_z']
velo_state_x = df['eminus_velostate_x']
velo_state_y = df['eminus_velostate_y']
velo_state_z = df['eminus_velostate_z']
energies = df['ECAL_cluster_e_arr']


e_cluster_x, e_cluster_y = find_electron([ex,ey], [clusters_x, clusters_y])



photons_x, photons_y, ph_sprx, ph_spry, ph_cl, ph_pt = move_photons([photon_clusters_x,photon_clusters_y], [clusters_x, clusters_y], df['ECAL_photon_sprx_arr'], df['ECAL_photon_spry_arr'], df['ECAL_photon_CL_arr'],df['ECAL_photon_PT_arr'])
if training_frame:
    daughter_clusters_x, daughter_clusters_y = find_daughter_clusters([daughters_x, daughters_y], [clusters_x, clusters_y], e_cluster_x, DAUGHTER_CLUSTER_MATCH_MAX)

# filter out cadidate clusters to feed into neural network
weighting = get_weights([ex, ey],[clusters_x, clusters_y], [line_x, line_y])
filtered_x, filtered_y, weighting, energy ,ph_sprx, ph_spry, ph_cl, ph_pt= filter_candidates([clusters_x, clusters_y], energies, weighting, n_candidate,ph_sprx, ph_spry, ph_cl, ph_pt)
if not training_frame:
    daughter_clusters_x = None

training_data = compose_df(df_backup, filtered_x, filtered_y, weighting, energy, daughter_clusters_x,ph_sprx, ph_spry, ph_cl, ph_pt, n_candidate)

training_data
training_data['labels']



model = XGBClassifier({'nthread':4}) #init model
model.load_model("C:/Users/felix/Documents/University/Thesis/BremsAdder_LHCb/classifier.model")
training_data['preds'] = model.predict(np.array(training_data.drop(['labels'], axis=1)))


true_brem = 0
false_Brem = 0
true_elephant = 0
false_elephant = 0
count=0
for i in range(len(df)):
    if df['eminus_BremMultiplicity'][i] > 0 and len(daughter_clusters_x[i])>0:
        plt.scatter(daughter_clusters_x[i], daughter_clusters_y[i], c='green', s=100)
        true_brem += 1
    elif len(daughter_clusters_x[i])>0 and df['eminus_BremMultiplicity'][i] == 0:
        plt.scatter(daughter_clusters_x[i], daughter_clusters_y[i], c='black', s=100)
        false_Brem += 1
    elif len(daughter_clusters_x[i]) == 0 and df['eminus_BremMultiplicity'][i] == 0:
        true_brem += 1
    else:
        false_Brem += 1



    for j in range(n_candidate):
        if training_data['preds'][count] == True:
            plt.scatter(training_data['x_pos'][count], training_data['y_pos'][count], c='orange', marker=1)
            if training_data['x_pos'][count] in daughter_clusters_x[i]:
                true_elephant+=1
            else:
                false_elephant+=1
        if training_data['preds'][count] == False:
            if training_data['x_pos'][count] in daughter_clusters_x[i]:
                false_elephant+=1
            else:
                true_elephant+=1

        count+=1

print("True_brem:", true_brem)
print("False brem:", false_Brem)

print("true_elephant", true_elephant)
print("false_elephant", false_elephant)

plt.show()
