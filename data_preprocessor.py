import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from itertools import compress
import xgboost as xgb
from sklearn.model_selection import train_test_split
try:
    from Code.xgboosting import *
except:
    from xgboosting import *

#%%

############# HELPER FUNCTIONS ######################


def compose_df(df, filtered_x, filtered_y, scaling_arr, weighting, p_clusters_x, filtered_energy, daughter_clusters_x):
    # df is the initial datafram loaded

    # compose dataframe based on results
    # Here, I will create a dataframe where each row is for a cluster and a frame with
    # labels on whether said deposit is a daughter photon to do advanced filtering. Then,
    # this filtered information will be fed into the final neural network to calculate
    # the energy
    frame = []
    for i in range(len(filtered_x)):
        for j in range(len(filtered_x[i])):
            frame.append(filtered_x[i][j])
    X_train = pd.DataFrame({"x_pos": frame})
    frame = []
    for i in range(len(filtered_x)):
        for j in range(len(filtered_x[i])):
            frame.append(filtered_y[i][j])
    X_train['y_pos']=frame
    scaling = []
    frame = []
    for i in range(len(filtered_x)):
        for j in range(len(filtered_x[i])):
            scaling.append(scaling_arr[i])
            frame.append(weighting[i][j])
    X_train['weighting'] = frame
    X_train['scaling'] = scaling

    is_phot = []
    for i in range(len(filtered_x)):
        for j in range(len(filtered_x[i])):
            is_phot.append(filtered_x[i][j] in p_clusters_x[i])
    X_train['is_photon'] = is_phot

    energies = []
    for i in range(len(filtered_x)):
        for j in range(len(filtered_x[i])):
            energies.append(filtered_energy[i][j])
    X_train['energy'] = energies
    if daughter_clusters_x is not None:
        labels = []
        for i in range(len(filtered_x)):
            for j in range(len(filtered_x[i])):
                labels.append(filtered_x[i][j] in daughter_clusters_x[i])
        X_train['labels'] = labels
    single_X_train = create_single_df(df)


    # make each row n_candidate times often in df
    pd_arr = pd.DataFrame(np.repeat(np.array(single_X_train), 10, axis=0))
    df1 = pd.DataFrame(data=pd_arr.values, columns=single_X_train.columns)

    training_data = pd.concat([X_train, df1], axis=1)

    return training_data

def create_single_df(df):
    single_X_train = pd.DataFrame({"eminus_nobrem_P": np.array(df['eminus_nobrem_P'])})
    single_X_train['eminus_nobrem_PT'] = df['eminus_nobrem_PT']
    single_X_train['eminus_OWNPV_X'] = df['eminus_OWNPV_X']
    single_X_train['eminus_OWNPV_Y'] = df['eminus_OWNPV_Y']
    single_X_train['eminus_OWNPV_Z'] = df['eminus_OWNPV_Z']
    single_X_train['eminus_OWNPV_XERR'] = df['eminus_OWNPV_XERR']
    single_X_train['eminus_OWNPV_YERR'] = df['eminus_OWNPV_YERR']
    single_X_train['eminus_OWNPV_ZERR'] = df['eminus_OWNPV_ZERR']
    single_X_train['eminus_OWNPV_CHI2'] = df['eminus_OWNPV_CHI2']
    single_X_train['eminus_OWNPV_NDOF'] = df['eminus_OWNPV_NDOF']
    single_X_train['eminus_IP_OWNPV'] = df['eminus_IP_OWNPV']
    single_X_train['eminus_IPCHI2_OWNPV'] = df['eminus_IPCHI2_OWNPV']
    single_X_train['eminus_ORIVX_X'] = df['eminus_ORIVX_X']
    single_X_train['eminus_ORIVX_Y'] = df['eminus_ORIVX_Y']
    single_X_train['eminus_ORIVX_Z'] = df['eminus_ORIVX_Z']
    single_X_train['eminus_ORIVX_XERR'] = df['eminus_ORIVX_XERR']
    single_X_train['eminus_ORIVX_YERR'] = df['eminus_ORIVX_YERR']
    single_X_train['eminus_ORIVX_ZERR'] = df['eminus_ORIVX_ZERR']
    single_X_train['eminus_ORIVX_CHI2'] = df['eminus_ORIVX_CHI2']
    single_X_train['eminus_ORIVX_NDOF'] = df['eminus_ORIVX_NDOF']
    single_X_train['eminus_ECAL_velotrack_x'] = df['eminus_ECAL_velotrack_x']
    single_X_train['eminus_ECAL_velotrack_y'] = df['eminus_ECAL_velotrack_y']
    single_X_train['eminus_ECAL_velotrack_sprx'] = df['eminus_ECAL_velotrack_sprx']
    single_X_train['eminus_ECAL_velotrack_spry'] = df['eminus_ECAL_velotrack_spry']
    single_X_train['eminus_ECAL_TTtrack_x'] = df['eminus_ECAL_TTtrack_x']
    single_X_train['eminus_ECAL_TTtrack_y'] = df['eminus_ECAL_TTtrack_y']
    single_X_train['eminus_ECAL_TTtrack_sprx'] = df['eminus_ECAL_TTtrack_sprx']
    single_X_train['eminus_ECAL_TTtrack_spry'] = df['eminus_ECAL_TTtrack_spry']
    single_X_train['eminus_P'] = df['eminus_P']
    single_X_train['eminus_PT'] = df['eminus_PT']
    single_X_train['eminus_PE'] = df['eminus_PE']
    single_X_train['eminus_PX'] = df['eminus_PX']
    single_X_train['eminus_PY'] = df['eminus_PY']
    single_X_train['eminus_PZ'] = df['eminus_PZ']
    single_X_train['eminus_M'] = df['eminus_M']
    single_X_train['eminus_ID'] = df['eminus_ID']
    single_X_train['eminus_PIDe'] = df['eminus_PIDe']
    single_X_train['eminus_PIDmu'] = df['eminus_PIDmu']
    single_X_train['eminus_PIDK'] = df['eminus_PIDK']
    single_X_train['eminus_PIDp'] = df['eminus_PIDp']
    single_X_train['eminus_PIDd'] = df['eminus_PIDd']
    single_X_train['eminus_ProbNNe'] = df['eminus_ProbNNe']
    single_X_train['eminus_ProbNNk'] = df['eminus_ProbNNk']
    single_X_train['eminus_ProbNNp'] = df['eminus_ProbNNp']
    single_X_train['eminus_ProbNNpi'] = df['eminus_ProbNNpi']
    single_X_train['eminus_ProbNNmu'] = df['eminus_ProbNNmu']
    single_X_train['eminus_ProbNNd'] = df['eminus_ProbNNd']
    single_X_train['eminus_ProbNNghost'] = df['eminus_ProbNNghost']
    single_X_train['eminus_hasMuon'] = df['eminus_hasMuon']
    single_X_train['eminus_isMuon'] = df['eminus_isMuon']
    single_X_train['eminus_hasRich'] = df['eminus_hasRich']
    single_X_train['eminus_UsedRichAerogel'] = df['eminus_UsedRichAerogel']
    single_X_train['eminus_UsedRich1Gas'] = df['eminus_UsedRich1Gas']
    single_X_train['eminus_UsedRich2Gas'] = df['eminus_UsedRich2Gas']
    single_X_train['eminus_RichAboveElThres'] = df['eminus_RichAboveElThres']
    single_X_train['eminus_RichAboveMuThres'] = df['eminus_RichAboveMuThres']
    single_X_train['eminus_RichAbovePiThres'] = df['eminus_RichAbovePiThres']
    single_X_train['eminus_RichAboveKaThres'] = df['eminus_RichAboveKaThres']
    single_X_train['eminus_RichAbovePrThres'] = df['eminus_RichAbovePrThres']
    single_X_train['eminus_hasCalo'] = df['eminus_hasCalo']
    single_X_train['eminus_TRACK_Type'] = df['eminus_TRACK_Type']
    single_X_train['eminus_TRACK_Key'] = df['eminus_TRACK_Key']
    single_X_train['eminus_TRACK_CHI2NDOF'] = df['eminus_TRACK_CHI2NDOF']
    single_X_train['eminus_TRACK_PCHI2'] = df['eminus_TRACK_PCHI2']
    single_X_train['eminus_TRACK_MatchCHI2'] = df['eminus_TRACK_MatchCHI2']
    single_X_train['eminus_TRACK_GhostProb'] = df['eminus_TRACK_GhostProb']
    single_X_train['eminus_TRACK_CloneDist'] = df['eminus_TRACK_CloneDist']
    single_X_train['eminus_TRACK_Likelihood'] = df['eminus_TRACK_Likelihood']
    single_X_train['nCandidate'] = df['nCandidate']
    single_X_train['totCandidates'] = df['totCandidates']
    single_X_train['EventInSequence'] = df['EventInSequence']
    single_X_train['N_ECAL_clusters'] = df['N_ECAL_clusters']
    single_X_train['N_ECAL_photons'] = df['N_ECAL_photons']
    single_X_train['BCID'] = df['BCID']
    single_X_train['BCType'] = df['BCType']
    single_X_train['OdinTCK'] = df['OdinTCK']
    single_X_train['L0DUTCK'] = df['L0DUTCK']
    single_X_train['HLT1TCK'] = df['HLT1TCK']
    single_X_train['HLT2TCK'] = df['HLT2TCK']
    single_X_train['GpsTime'] = df['GpsTime']
    single_X_train['Polarity'] = df['Polarity']
    single_X_train['nPV'] = df['nPV']
    return single_X_train

def plot_graph(overlay = True, clusters_x=None, clusters_y=None, daughters_x=None,
                daughters_y=None, ex=None, ey=None, line_x=None, line_y=None,
                electron_cluster_x=None, electron_cluster_y=None, daughter_clusters_x=None,
                daughter_clusters_y=None, photon_clusters_x=None, photon_clusters_y=None):
    try:
        for i in range(min(len(df), N_PLOTS)):
            try:
                plt.scatter(daughters_x[i], daughters_y[i], c='orange',s=60, marker='o')#
            except:
                pass
            try:
                plt.scatter(clusters_x[i], clusters_y[i], c='black', s=30, marker='o')
            except:
                pass
            try:
                plt.scatter(ex[i], ey[i], c='blue',s=60, marker='3')
            except:
                pass
            try:
                plt.scatter(electron_cluster_x[i], electron_cluster_y[i], c='yellow',s=60, marker='o')
            except:
                pass
            try:
                plt.scatter(daughter_clusters_x[i], daughter_clusters_y[i], c='orange',s=60, marker='3')
            except:
                pass
            try:
                plt.scatter(photon_clusters_x[i], photon_clusters_y[i], c='blue',s=20, marker='x')
            except:
                pass
            try:
                plt.plot([0, line_x[i]], [0, line_y[i]], c='blue')
            except:
                pass
            if overlay==False:
                try:
                    plt.title(df['eminus_HasBremAdded'][i] + "Evenet Number " + str(i))
                    axes = plt.gca()
                    axes.set_xlim([-4000, 4000])
                    axes.set_ylim([-4000, 4000])
                    plt.show()
                except:
                    pass
        if overlay:
            try:
                axes = plt.gca()
                axes.set_xlim([-4000, 4000])
                axes.set_ylim([-4000, 4000])
                plt.show()
            except:
                pass
    except:
        print("The plotting isn't working - please write a custom function")

'''finds distance between two points'''
def get_distance(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)




'''centers values around array of center points'''
def center_values(center, data_points):
    return np.subtract(center,data_points)

'''rotates array around center points to make an array of points (line_end) be rotated to lie on the x-axis'''
def rotate(line_end, data):
    line_x, line_y = line_end
    x, y = data
    radians = -np.arcsin(line_y/get_distance([0,0], [line_x, line_y]))
    xx = x * np.cos(radians) + y * np.sin(radians)
    yy = (y/get_distance([0,0], [x, y])+np.sin(radians))*get_distance([0,0], [x, y])
    return xx, yy

'''moves the photons identifications to their corresponding clusters'''
def move_photons(photons, clusters):
    clusters_x, clusters_y = clusters
    photons_x, photons_y = photons
    for i in range(len(clusters_x)):
        for j in range(len(photons_x[i])):
            dist = get_distance([photons_x[i][j], photons_y[i][j]], [clusters_x[i], clusters_y[i]])
            minimum = np.argmin(dist)
            photons_x[i][j] = clusters_x[i][minimum]
            photons_y[i][j] = clusters_y[i][minimum]
    return photons_x, photons_y

'''normalizes (zooms) datapoints'''
def norm(line_end, data):
    # normalized distances
    normalization = (1/(line_end[0]-[0,0][0]))*2000
    return data[0]*normalization, data[1]*normalization

'''returns how strongly an image was zoomed'''
def get_scaling(line_end):
    return np.abs((1/(line_end[0]-[0,0][0]))*2000)


'''finds cluster closest to the projected electron position'''
def find_electron(projection, clusters):
    clusters_x, clusters_y = clusters
    electron_cluster_x = np.ndarray(len(clusters_x))
    electron_cluster_y = np.ndarray(len(clusters_y))
    for i in range(len(clusters_x)):
        dist = get_distance([0,0], [clusters_x[i], clusters_y[i]])
        minimum = np.argmin(dist)
        # print(minimum)
        electron_cluster_x[i] = clusters_x[i][minimum]
        # print(electron_cluster_x[i])
        electron_cluster_y[i] = clusters_y[i][minimum]
    return electron_cluster_x, electron_cluster_y


'''finds daughter clusters based on where the mctruth information predicts daughter photons'''
def find_daughter_clusters(daughters, clusters, scaling, electron_cluster_x, DAUGHTER_CLUSTER_MATCH_MAX):
    daughter_clusters_x = []
    daughter_clusters_y = []
    clusters_x, clusters_y = clusters
    daughters_x, daughters_y = daughters
    for i in range(len(clusters_x)):
        daughter_clusters_x.append([])
        daughter_clusters_y.append([])
        for j in range(len(daughters_x[i])):
            dist = get_distance([daughters_x[i][j],daughters_y[i][j]],[clusters_x[i],clusters_y[i]])
            if np.min(dist)/scaling[i] < DAUGHTER_CLUSTER_MATCH_MAX:
                minimum = np.argmin(dist)
                if clusters_x[i][minimum] != electron_cluster_x[i]:
                    daughter_clusters_x[-1].append([clusters_x[i][minimum]])
                    daughter_clusters_y[-1].append([clusters_y[i][minimum]])
    return daughter_clusters_x, daughter_clusters_y

'''gives each cluster a weighted (geometric distances from center)'''
def get_weights(clusters, line_end, scaling_arr):
    # finds distance from line
    clusters_x, clusters_y = clusters
    line_x, line_y = line_end
    dist_line = []
    dist_point = []
    weight = 1
    for i in range(len(clusters_x)):
        p1 = np.array([0,0])
        p2 = np.array([line_x[i],line_y[i]])
        p3 = []
        for x,y in zip(clusters_x[i], clusters_y[i]):
            p3.append([x,y])
        p3 = np.array(p3)
        dist_line.append(abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))/scaling_arr[i])
    for i in range(len(clusters_x)):
        d1 = np.abs(get_distance([line_x[i], line_y[i]], [clusters_x[i], clusters_y[i]]))
        d2 = np.abs(get_distance([0,0], [clusters_x[i], clusters_y[i]]))
        dist_point.append((d1+d2)*weight)
    return np.array(dist_line)+np.array(dist_point)


'''filters candidate clusters based weighted distance'''
def filter_candidates(data, energy, weights, number_candidates):
    data_x, data_y = data
    return_x = np.ndarray((len(data_x),number_candidates))
    return_y = np.ndarray((len(data_y),number_candidates))
    return_weights = np.ndarray((len(weights), number_candidates))
    return_energy = np.ndarray((len(energy), number_candidates))
    for i in range(len(data_x)):
        s = sorted(zip(weights[i], data_x[i]))
        w, sor_x = map(list, zip(*s))
        s = sorted(zip(weights[i], data_y[i]))
        w, sor_y = map(list, zip(*s))
        s = sorted(zip(weights[i], energy[i]))
        w, ener = map(list, zip(*s))
        # if <number_candidates deposits stock up with very far away Deposits
        for i in range(number_candidates-len(sor_x)):
            sor_x.append(10**30)
            sor_y.append(10**30)
            w.append(10**30)
            ener.append(-10**30)
        return_x[i] = sor_x[:number_candidates]
        return_y[i] = sor_y[:number_candidates]
        return_weights[i] = w[:number_candidates]
        return_energy[i] = ener[:number_candidates]
    return return_x, return_y, return_energy, return_weights

'''this function returns a refined filter using xgboost (functino in XGB_Filter)'''
def assign_prob(model, data):
    return model.predict_proba(data)

def predict(model, data):
    return model.predict(data)

def arr_filter(x, bool_list):
    return list(compress(x, bool_list))

def fine_filter(preds, clusters):
    clusters_x, clusters_y = clusters
    return arr_filter(clusters_x, preds), arr_filter(clusters_y, preds)


#%%


######################### MAIN FUNCTIONS ######################################
'''Creates a dataframe to train an ml algorithm to identify photon clusters'''
def return_candidate_frame(df, DAUGHTER_CLUSTER_MATCH_MAX=75, n_candidate = 10):
    # set variables
    training_frame = False # specifies whether frame can be used for training
    df_backup = copy.deepcopy(df)
    ex = df['eminus_ECAL_x']
    ey = df['eminus_ECAL_y']
    clusters_x = df['ECAL_cluster_x_arr']
    clusters_y = df['ECAL_cluster_y_arr']
    try:
        daughters_x = df['eminus_MCphotondaughters_ECAL_X']
        daughters_y = df['eminus_MCphotondaughters_ECAL_Y']
        training_frame = True
    except:
        print("This frame cannot be used for training - only prediction")
        daughters_x=None
        daughters_y=None
        training_frame = False
    photon_clusters_x = df['ECAL_photon_x_arr']
    photon_clusters_y = df['ECAL_photon_y_arr']
    velo_x = df['eminus_ECAL_velotrack_x']
    velo_y = df['eminus_ECAL_velotrack_y']
    ttrack_x = df['eminus_ECAL_TTtrack_x']
    ttrack_y = df['eminus_ECAL_TTtrack_y']
    energies = df['ECAL_cluster_e_arr']
    line_x, line_y = (velo_x+ttrack_x)/2, (velo_y+ttrack_y)/2
    center = [ex, ey]

    # center datapoints around electron
    centered_clusters_x, centered_clusters_y = center_values(center, [clusters_x, clusters_y])
    if training_frame:
        centered_daughters_x, centered_daughters_y = center_values(center, [daughters_x, daughters_y])
    centered_velo_x, centered_velo_y = center_values(center,[velo_x, velo_y])
    centered_ttrack_x, centered_ttrack_y = center_values(center, [ttrack_x, ttrack_y])
    centered_line_x, centered_line_y = center_values(center, [line_x, line_y])
    centered_photon_clusters_x, centered_photon_clusters_y = center_values(center, [photon_clusters_x, photon_clusters_y])

    # move to photons to their corresponding clusters
    centered_photon_clusters_x, centered_photon_clusters_y = move_photons([centered_photon_clusters_x, centered_photon_clusters_y], [centered_clusters_x, centered_clusters_y])
    # plot_graph(photon_clusters_x=centered_photon_clusters_x, photon_clusters_y=centered_photon_clusters_y, clusters_x=centered_clusters_x, clusters_y=centered_clusters_y)

    # rotate
    line_end = [centered_line_x, centered_line_y]
    rot_cen_clusters_x, rot_cen_clusters_y = copy.deepcopy(centered_clusters_x), copy.deepcopy(centered_clusters_y)
    if training_frame:
        rot_cen_daughters_x, rot_cen_daughters_y = copy.deepcopy(centered_daughters_x), copy.deepcopy(centered_clusters_y)
    rot_cen_photon_clusters_x, rot_cen_photon_clusters_y = copy.deepcopy(centered_photon_clusters_x),copy.deepcopy(centered_photon_clusters_y)

    for i in range(len(centered_clusters_x)):
        rot_cen_clusters_x[i], rot_cen_clusters_y[i] = rotate([centered_line_x[i], centered_line_y[i]], [centered_clusters_x[i], centered_clusters_y[i]])
        if training_frame:
            rot_cen_daughters_x[i], rot_cen_daughters_y[i] = rotate([centered_line_x[i], centered_line_y[i]],[centered_daughters_x[i], centered_daughters_y[i]])
        rot_cen_photon_clusters_x[i], rot_cen_photon_clusters_y[i] = rotate([centered_line_x[i], centered_line_y[i]], [centered_photon_clusters_x[i], centered_photon_clusters_y[i]])
    rot_cen_line_x, rot_cen_line_y = rotate([centered_line_x, centered_line_y],[centered_line_x, centered_line_y])

    # plot_graph(clusters_x=rot_cen_clusters_x, clusters_y=rot_cen_clusters_y, photon_clusters_x=rot_cen_photon_clusters_x, photon_clusters_y=rot_cen_photon_clusters_y)


    # normalize
    for i in range(len(centered_clusters_x)):
        rot_cen_clusters_x[i], rot_cen_clusters_y[i] = norm([rot_cen_line_x[i],rot_cen_line_y[i]], [rot_cen_clusters_x[i], rot_cen_clusters_y[i]])
        if training_frame:
            rot_cen_daughters_x[i], rot_cen_daughters_y[i] = norm([rot_cen_line_x[i],rot_cen_line_y[i]],[rot_cen_daughters_x[i], rot_cen_daughters_y[i]])
        rot_cen_photon_clusters_x[i], rot_cen_photon_clusters_y[i] = norm([centered_line_x[i], centered_line_y[i]], [rot_cen_photon_clusters_x[i], rot_cen_photon_clusters_y[i]])
    rot_cen_line_x, rot_cen_line_y = norm([rot_cen_line_x,rot_cen_line_y],[rot_cen_line_x, rot_cen_line_y])

    # plot_graph(clusters_x=rot_cen_clusters_x, clusters_y=rot_cen_clusters_y, photon_clusters_x=rot_cen_photon_clusters_x, photon_clusters_y=rot_cen_photon_clusters_y)


    # find electron clusters based on mc truth information

    e_cluster_x, e_cluster_y = find_electron([0,0], [rot_cen_clusters_x, rot_cen_clusters_y])
    scaling_arr = get_scaling(line_end)

    if training_frame:
        daughter_clusters_x, daughter_clusters_y = find_daughter_clusters([rot_cen_daughters_x, rot_cen_daughters_y], [rot_cen_clusters_x, rot_cen_clusters_y], scaling_arr, e_cluster_x, DAUGHTER_CLUSTER_MATCH_MAX)

    # filter out cadidate clusters to feed into neural network

    weighting = get_weights([rot_cen_clusters_x, rot_cen_clusters_y], [rot_cen_line_x, rot_cen_line_y], scaling_arr)
    filtered_x, filtered_y, weighting, energy = filter_candidates([rot_cen_clusters_x, rot_cen_clusters_y], energies, weighting, n_candidate)

    if not training_frame:
        daughter_clusters_x = None
    training_data = compose_df(df_backup, filtered_x, filtered_y, scaling_arr, weighting, rot_cen_photon_clusters_x, energy, daughter_clusters_x)

    return training_data

'''returns a dataframe based on the return_candidate_frame function that can be used to train a ml algorithm to find the energy of an electron'''
def return_calculation_frame(model, cand_frame, orig_df, n_candidate=10):
    assert len(cand_frame) == len(orig_df)*n_candidate
    try:
        cand_frame = cand_frame.drop(['labels'], axis=1)
    except:
        pass
    # preds = predict(model, np.array(cand_frame))
    # final_x, final_y = fine_filter(preds, [cand_frame['x_pos'], cand_frame['y_pos']])
    probs = assign_prob(model, np.array(cand_frame))
    photon_data = pd.DataFrame({"xgb_prob_bkgnd": probs.T[0]})
    photon_data['xgb_prob_daughter'] = probs.T[1]
    photon_data['x_pos'] = cand_frame['x_pos']
    photon_data['y_pos'] = cand_frame['y_pos']
    photon_data['weighting'] = cand_frame['weighting']
    photon_data['energy'] = cand_frame['energy']
    photon_data['is_photon'] = cand_frame['is_photon']
    # 'squish' all 10 candidate photons into a single row
    np_p_data = np.array(photon_data)
    np_p_data = np.reshape(np_p_data, (len(photon_data)//n_candidate,len(np_p_data[0])*n_candidate))
    assert len(np_p_data) == len(orig_df)
    p_concat = pd.DataFrame(np_p_data)
    # add labels
    TRUE_P = np.sqrt(orig_df['eminus_TRUEP_X']**2 + orig_df['eminus_TRUEP_Y']**2 + orig_df['eminus_TRUEP_Z']**2)
    training_concat = create_single_df(orig_df)
    training_concat['labels'] = TRUE_P
    assert len(training_concat) == len(orig_df)
    training_NN = pd.concat([p_concat, training_concat], axis=1, sort=False)
    return training_NN


#%%
###################################### MAIN ################################
filename = "C:/Users/felix/Documents/University/Thesis/final_large_electron_set"
MAX_ROWS = 100
df = pd.read_pickle(filename)
df = df[:MAX_ROWS]
# df = df.drop(['eminus_MCphotondaughters_ECAL_X'], axis=1)
# df = df.drop(['eminus_MCphotondaughters_ECAL_Y'], axis=1)
cand_frame = return_candidate_frame(df)
y = np.array(cand_frame['labels'])
X = np.array(cand_frame.drop(['labels'], axis=1))
X_train, X_test, y_train, y_test = train_test_split(X,y)
model = train_classifier(X=X_train, y=y_train)
# get_metrics(model, X_test, y_test)


# make predictiions for whole dataset
# preds = predict(model, np.array(cand_frame.drop(['labels'], axis=1)))
# probs = assign_prob(model, np.array(cand_frame.drop(['labels'], axis=1)))
probs = assign_prob(model, np.array(cand_frame.drop(['labels'], axis=1)))
probs.shape
cand_frame.shape
df.shape
calc_frame = return_calculation_frame(model, cand_frame, df)
calc_frame





















#%%



# training_data
# training_data.to_csv("C:/Users/felix/Documents/University/Thesis/training_data_csv")


#%%
# train xgboost FIIIINALLY SOME ML





#%%
# JUST FOR VISUALIZATION
# apply fine filtering to the roughly filtered values
# flat_x, flat_y = [], []
# for i in range(len(filtered_x)):
#     for j in range(len(filtered_x[i])):
#         flat_x.append(filtered_x[i][j])
#         flat_y.append(filtered_y[i][j])
#
# len(flat_x)
# final_x, final_y = fine_filter(preds, [flat_x, flat_y])
# len(final_x)
#
# get_metrics(model, X_test, y_test)


# np.save("C:/Users/felix/Documents/University/Thesis/BremsAdder_LHCb/Files/filtered_clusters", [final_x, final_y])
# np.save("C:/Users/felix/Documents/University/Thesis/BremsAdder_LHCb/Files/daughter_clusters", [daughter_clusters_x, daughter_clusters_y])
# MAX = 10000000
# plt.scatter(final_x[:MAX], final_y[:MAX], c='black')
# for i in range(min(len(daughter_clusters_x),MAX)):
#     plt.scatter(daughter_clusters_x[i], daughter_clusters_y[i], c='orange', marker='3')
# axes = plt.gca()
# axes.set_xlim([-4000, 4000])
# axes.set_ylim([-4000, 4000])
# plt.show()
# plot_graph(clusters_x=final_x, clusters_y=final_y, daughter_clusters_x=daughter_clusters_x, daughter_clusters_y=daughter_clusters_y)
#%%
################### CREATE TRAINING DATA FOR NEURAL NETWORK ###################

# probs
# # create df with all data about specific photons
# photon_data = pd.DataFrame({"xgb_prob_bkgnd": probs.T[0]})
# photon_data['xgb_prob_daughter'] = probs.T[1]
# photon_data['x_pos'] = training_data['x_pos']
# photon_data['y_pos'] = training_data['y_pos']
# photon_data['weighting'] = training_data['weighting']
# photon_data['energy'] = training_data['energy']
# photon_data['is_photon'] = training_data['is_photon']
#
# # 'squish' all 10 candidate photons into a single row
# np_p_data = np.array(photon_data)
# np_p_data = np.reshape(np_p_data, (-1,len(np_p_data[0])*n_candidate))
# p_concat = pd.DataFrame(np_p_data)
#
# df = pd.read_pickle(filename)
# df = df[:MAX_ROWS]
# training_concat = create_single_df(df)
# TRUE_P = np.sqrt(df['eminus_TRUEP_X']**2 + df['eminus_TRUEP_Y']**2 + df['eminus_TRUEP_Z']**2)
# training_concat['labels'] = TRUE_P
# plt.hist(TRUE_P)
# training_NN = pd.concat([p_concat, training_concat], axis=1, sort=False)
# training_NN.shape
# training_NN.to_csv("C:/Users/felix/Documents/University/Thesis/training_NNv2")
# pd.read_csv("C:/Users/felix/Documents/University/Thesis/training_NNv2")
# plot_graph(overlay=True, clusters_x=rot_cen_clusters_x, clusters_y=rot_cen_clusters_y,
#             daughters_x=rot_cen_daughters_x, daughters_y=rot_cen_daughters_y,
#             line_x=rot_cen_line_x, line_y=rot_cen_line_y, electron_cluster_x=e_cluster_x,
#             electron_cluster_y=e_cluster_y, daughter_clusters_x=daughter_clusters_x,
#             daughter_clusters_y=daughter_clusters_y, photon_clusters_x=rot_cen_photon_clusters_x,
#             photon_clusters_y=rot_cen_photon_clusters_y)

#%%
# create histogram of how many daughter clusters there are
# histogram = []
# for i in range(len(daughter_clusters_x)):
#     histogram.append((len(daughter_clusters_x[i])))
#
# return_x = np.ndarray((len(rot_cen_clusters_x),20))
# return_x[0].shape
#
# plt.hist(df['eminus_MCphotondaughters_N'])
# plt.hist(histogram, bins=11, range=(0,11))
# plt.hist(df['eminus_MCphotondaughters_N']-histogram, bins=11,range=(0,11))
