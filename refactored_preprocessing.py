import pandas as pd
import numpy as np
import copy
try:
    from Code.track_reconstruction import *
    from Code.xgboosting import *
except:
    try:
        from xgboosting import *
        from data_preprocessor import *
        from track_reconstruction import *
    except:
        from C.Users.szymon.Documents.xgboosting import *
'''This creates a new dataframe containing only the values that can be used for
predictions and are not arrays'''
def get_non_arrays(df):
    # so those are the values that are a single value per electron,
    # meaning that clusters (multiple per electron) etc. are not in here
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
    single_X_train['eminus_ECAL_x'] = df['eminus_ECAL_x']
    single_X_train['eminus_ECAL_y'] = df['eminus_ECAL_y']

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
    try:
        single_X_train['eminus_TTstate_x'] = df['eminus_TTstate_x']
        single_X_train['eminus_TTstate_y'] = df['eminus_TTstate_y']
        single_X_train['eminus_TTstate_z'] = df['eminus_TTstate_z']
        single_X_train['eminus_velostate_x'] = df['eminus_velostate_x']
        single_X_train['eminus_velostate_y'] = df['eminus_velostate_y']
        single_X_train['eminus_velostate_z'] = df['eminus_velostate_z']
    except:
        print("Note that you are using a dataset on which no track reconstruction can be performed")
    return single_X_train

'''finds distance between two points'''
def get_distance(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

# finds the cluster that belongs to the electron
def find_electron(projection, clusters):
    ex, ey = projection
    clusters_x, clusters_y = clusters
    electron_cluster_x = np.ndarray(len(clusters_x))
    electron_cluster_y = np.ndarray(len(clusters_y))
    for i in range(len(clusters_x)):
        dist = get_distance([ex[i], ey[i]], [clusters_x[i], clusters_y[i]])
        minimum = np.argmin(dist)
        # print(minimum)
        electron_cluster_x[i] = clusters_x[i][minimum]
        # print(electron_cluster_x[i])
        electron_cluster_y[i] = clusters_y[i][minimum]
    return electron_cluster_x, electron_cluster_y

#TODO: Find out why cluster_x[i] is sometimes longer than ph_spr[i]
'''move photons to their corresponding cluster'''
def move_photons(photons, clusters, ph_sprx, ph_spry, ph_cl, ph_pt):
    # The photon clusters don't _exactly_ overlap with the clusters - just roughly
    # Here we move them on top of the clusters
    clusters_x, clusters_y = clusters
    photons_x, photons_y = photons
    photon_sprx = []
    photon_spry = []
    photon_CL = []
    photon_PT = []
    for i in range(len(clusters_x)):
        photon_sprx.append([])
        photon_spry.append([])
        photon_CL.append([])
        photon_PT.append([])
        for j in range(len(photons_x[i])):
            dist = get_distance([photons_x[i][j], photons_y[i][j]], [clusters_x[i], clusters_y[i]])
            minimum = np.argmin(dist)
            photons_x[i][j] = clusters_x[i][minimum]
            photons_y[i][j] = clusters_y[i][minimum]
        count = 0
        try:
            for k in range(len(clusters_x[i])):
                # if count>=
                # print(clusters_x[i][k], photons_x[i])
                if  clusters_x[i][k] in photons_x[i]:
                    photon_sprx[i].append(ph_sprx[i][count])
                    photon_spry[i].append(ph_spry[i][count])
                    photon_CL[i].append(ph_cl[i][count])
                    photon_PT[i].append(ph_pt[i][count])
                    count+=1
                else:
                    # print('Entered')
                    photon_sprx[i].append(-1)
                    photon_spry[i].append(-1)
                    photon_CL[i].append(-1)
                    photon_PT[i].append(-1)
        except:
            print(i)
            print(count)
            print(len(ph_sprx[i]))
            print(len(clusters_x[i]))
    return photons_x, photons_y, photon_sprx, photon_spry, photon_CL, photon_PT

'''finds daughter clusters based on where the mctruth information predicts daughter photons'''
def find_daughter_clusters(daughters, clusters, electron_cluster_x, DAUGHTER_CLUSTER_MATCH_MAX):
    # some daughter photons aren't measured and they aren't measured in the same place where the
    # daughter photons arrive. Therefore we try to identify if a cluster was caused by a
    # daughterphoton in this function
    daughter_clusters_x = []
    daughter_clusters_y = []
    clusters_x, clusters_y = clusters
    daughters_x, daughters_y = daughters
    for i in range(len(clusters_x)):
        daughter_clusters_x.append([])
        daughter_clusters_y.append([])
        for j in range(len(daughters_x[i])):
            dist = get_distance([daughters_x[i][j],daughters_y[i][j]],[clusters_x[i],clusters_y[i]])
            if np.min(dist) < DAUGHTER_CLUSTER_MATCH_MAX:
                minimum = np.argmin(dist)
                if clusters_x[i][minimum] != electron_cluster_x[i]:
                    daughter_clusters_x[-1].append([clusters_x[i][minimum]])
                    daughter_clusters_y[-1].append([clusters_y[i][minimum]])
    return daughter_clusters_x, daughter_clusters_y

'''gives each cluster a weighted (geometric distances from center)'''
def get_weights(electron, clusters, line_end):
    # this functino assigns each cluster a weight based on the geometrical distance
    # from ttrack. This can probably still be improved upon
    ex, ey = electron
    clusters_x, clusters_y = clusters
    line_x, line_y = line_end
    dist_line = []
    dist_point = []
    weight = 1
    # for i in range(len(clusters_x)):
    #     p1 = np.array([ex[i], ey[i]])
    #     p2 = np.array([line_x[i],line_y[i]])
    #     p3 = []
    #     for x,y in zip(clusters_x[i], clusters_y[i]):
    #         p3.append([x,y])
    #     p3 = np.array(p3)
    #     dist_line.append(abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)))
    for i in range(len(clusters_x)):
        d1 = np.abs(get_distance([line_x[i], line_y[i]], [clusters_x[i], clusters_y[i]]))
        # d2 = np.abs(get_distance([ex[i], ey[i]], [clusters_x[i], clusters_y[i]]))
        # dist_point.append((d1+d2)*weight)
        dist_point.append(d1)
    # return np.array(dist_line)+np.array(dist_point)
    return np.array(dist_point)


'''filters candidate clusters based weighted distance'''
def filter_candidates(data, energy, weights, number_candidates,  ph_sprx, ph_spry, ph_cl, ph_pt):
    # this function simply sorts the arrays based on their weighting and selects the
    # 10 closest ones
    data_x, data_y = data
    assert len(data_x)==len(energy)
    assert len(weights)==len(energy)
    return_x = np.ndarray((len(data_x),number_candidates))
    return_y = np.ndarray((len(data_y),number_candidates))
    return_weights = np.ndarray((len(weights), number_candidates))
    return_energy = np.ndarray((len(energy), number_candidates))
    return_ph_sprx = np.ndarray((len(ph_sprx), number_candidates))
    return_ph_spry = np.ndarray((len(ph_spry), number_candidates))
    return_ph_cl = np.ndarray((len(ph_cl), number_candidates))
    return_ph_pt = np.ndarray((len(ph_pt), number_candidates))


    for i in range(len(data_x)):
        s = sorted(zip(weights[i], data_x[i]))
        w, sor_x = map(list, zip(*s))
        s = sorted(zip(weights[i], data_y[i]))
        w, sor_y = map(list, zip(*s))
        s = sorted(zip(weights[i], energy[i]))
        w, ener = map(list, zip(*s))
        s = sorted(zip(weights[i], ph_sprx[i]))
        w, sprx = map(list, zip(*s))
        s = sorted(zip(weights[i], ph_spry[i]))
        w, spry = map(list, zip(*s))
        s = sorted(zip(weights[i], ph_cl[i]))
        w, cl = map(list, zip(*s))
        s = sorted(zip(weights[i], ph_pt[i]))
        w, pt = map(list, zip(*s))
        # if <number_candidates deposits stock up with very far away Deposits
        for i in range(number_candidates-len(sor_x)):
            sor_x.append(10**30)
            sor_y.append(10**30)
            w.append(10**30)
            ener.append(-10**30)
            sprx.append(10**30)
            spry.append(10**30)
            cl.append(10**30)
            pt.append(10**30)

        return_x[i] = sor_x[:number_candidates]
        return_y[i] = sor_y[:number_candidates]
        return_weights[i] = w[:number_candidates]
        return_energy[i] = ener[:number_candidates]
        return_ph_sprx[i] = sprx[:number_candidates]
        return_ph_spry[i] = spry[:number_candidates]
        return_ph_cl[i] = cl[:number_candidates]
        return_ph_pt[i] = pt[:number_candidates]

    return return_x, return_y, return_energy, return_weights, return_ph_sprx, return_ph_spry, return_ph_cl, return_ph_pt


def compose_df(df, filtered_x, filtered_y, weighting, energy, daughter_clusters_x,ph_sprx, ph_spry, ph_cl, ph_pt, n_candidate):
    # df is the initial datafram loaded

    # Here, I will create a dataframe where each row is for a cluster and a frame with
    # labels on whether said deposit is a daughter photon to do advanced filtering. Then,
    # this filtered information will be fed into the xgbooster
    x_pos = []
    y_pos = []
    weight = []
    cluster_sprx = []
    cluster_spry = []
    cluster_cl = []
    cluster_pt = []
    energies = []

    for i in range(len(filtered_x)):
        for j in range(len(filtered_x[i])):
            x_pos.append(filtered_x[i][j])
            y_pos.append(filtered_y[i][j])
            weight.append(weighting[i][j])
            cluster_sprx.append(ph_sprx[i][j])
            cluster_spry.append(ph_spry[i][j])
            cluster_cl.append(ph_cl[i][j])
            cluster_pt.append(ph_pt[i][j])
            energies.append(energy[i][j])

    X_train = pd.DataFrame({"x_pos": x_pos})
    X_train['weighting'] = weight
    X_train['cluster_sprx'] = cluster_sprx
    X_train['y_pos']=y_pos
    X_train['cluster_cl'] = cluster_cl
    X_train['cluster_spry'] = cluster_spry
    X_train['cluster_pt'] = cluster_pt
    X_train['energy'] = energies


    if daughter_clusters_x is not None:
        labels = []
        for i in range(len(filtered_x)):
            for j in range(len(filtered_x[i])):
                labels.append(filtered_x[i][j] in daughter_clusters_x[i])
        X_train['labels'] = labels
    single_X_train = get_non_arrays(df)





    # make each row n_candidate times often in df
    pd_arr = pd.DataFrame(np.repeat(np.array(single_X_train), n_candidate, axis=0))
    df1 = pd.DataFrame(data=pd_arr.values, columns=single_X_train.columns)

    training_data = pd.concat([X_train, df1], axis=1)
    # return X_train
    return training_data

'''Creates a dataframe to train an ml algorithm to identify photon clusters'''
def return_candidate_frame(df, DAUGHTER_CLUSTER_MATCH_MAX=75, n_candidate = 10):
    # returns a dataframe where each row corresponds to one candidate cluster

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
    line_x, line_y = (velo_x+ttrack_x)/2, (velo_y+ttrack_y)/2
    try:
        ttrack_state_x = df['eminus_TTstate_x']
        ttrack_state_y = df['eminus_TTstate_y']
        ttrack_state_z = df['eminus_TTstate_z']
        velo_state_x = df['eminus_velostate_x']
        velo_state_y = df['eminus_velostate_y']
        velo_state_z = df['eminus_velostate_z']
    except:
        print("This dataframe has no track information")
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

    return training_data

#%%

################## SECOND DATAFRAME ##################################

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
    photon_data['cluster_sprx'] = cand_frame['cluster_sprx']
    photon_data['cluster_spry'] = cand_frame['cluster_spry']
    photon_data['cluster_cl'] = cand_frame['cluster_cl']
    photon_data['cluster_pt'] = cand_frame['cluster_pt']
    photon_data['energy'] = cand_frame['energy']

    origins = []
    for i in range(len(cand_frame)):
        # get track for each event
        if i%n_candidate == 0:
            track = reconstruct_track(cand_frame, start=i, end=i+1)
            projection = project_track(cand_frame, track[0])
        origins.append(find_origin(track, projection, [cand_frame['x_pos'][i], cand_frame['y_pos'][i]]))
    origins = np.array(origins)

    photon_data['ph_origin_x'] = origins[:,0][:,0]
    photon_data['ph_origin_y'] = origins[:,1][:,0]
    photon_data['ph_origin_z'] = origins[:,2][:,0]


    # 'squish' all 10 candidate photons into a single row
    np_p_data = np.array(photon_data)
    np_p_data = np.reshape(np_p_data, (len(photon_data)//n_candidate,len(np_p_data[0])*n_candidate))
    assert len(np_p_data) == len(orig_df)
    p_concat = pd.DataFrame(np_p_data)
    # add labels
    TRUE_P = np.sqrt(orig_df['eminus_TRUEP_X']**2 + orig_df['eminus_TRUEP_Y']**2 + orig_df['eminus_TRUEP_Z']**2)
    # TRUE_P = orig_df['eminus_P']
    training_concat = get_non_arrays(orig_df)
    training_concat['labels'] = TRUE_P
    assert len(training_concat) == len(orig_df)
    training_NN = pd.concat([p_concat, training_concat], axis=1, sort=False)


    return training_NN

######### MAIN #############
#
#
#
# #
# import pandas as pd
# import numpy as np
# import copy
#
# # READ DATAFRAME
# try:
#     filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set"
#     df = pd.read_pickle(filename)
# except:
#     # model.load_model("C:/Users/felix/Documents/University/Thesis/BremsAdder_LHCb/classifier.model")
#     filename = "C:/Users/szymon/Documents/track_electron_set"
#     df = pd.read_pickle(filename)
#
# #
# START_ROW = 0
# MAX_ROWS = 20000
# df_orig = pd.read_pickle(filename)
# df_orig = df[START_ROW:MAX_ROWS+START_ROW]
# len(df_orig)
# df_orig = df.reset_index()
# # pd.set_option('display.max_columns', 1000)
# # print(df.columns.tolist())
# # # df_orig.columns.tolist()
# #
# cand_frame = return_candidate_frame(df_orig, DAUGHTER_CLUSTER_MATCH_MAX=100, n_candidate = 10)
# # cand_frame.columns
# # np.sum(df_orig['eminus_MCphotondaughters_N'])
# # np.sum(cand_frame.labels*1)
# # cand_frame.columns
# # # print(df['ECAL_cluster_x_arr'][0])
# #
#
#
#
# model = XGBClassifier({'nthread':4}) #init model
# regressor = XGBRegressor({'nthread':4}) # load data
# # model.load_model("C:/Users/felix/Documents/University/Thesis/BremsAdder_LHCb/classifier.model")
# # regressor.load_model("C:/Users/felix/Documents/University/Thesis/BremsAdder_LHCb/regressor.model")
# y = np.array(cand_frame['labels'])
# X = np.array(cand_frame.drop(['labels'], axis=1))
# model = train_classifier(X=X, y=y)
# probs = assign_prob(model, np.array(cand_frame.drop(['labels'], axis=1)))

# cand_frame.shape
#
# # frame = cand_frame.drop(['labels'], axis=1)
# calc_frame = return_calculation_frame(model, cand_frame, df)
# # import matplotlib.pyplot as plt
# # # ECAL_cluster_x_arr	ECAL_cluster_y_arr	ECAL_cluster_e_arr	N_ECAL_photons	ECAL_photon_x_arr
# # # ECAL_photon_y_arr	ECAL_photon_sprx_arr	ECAL_photon_spry_arr	ECAL_photon_CL_arr	ECAL_photon_PT_arr
# #
# # # plt.scatter(df['ECAL_photon_x_arr'][0],df['ECAL_photon_y_arr'][0], c='red', marker='x')
# # count = 0
# # for i in range(20):
# #     plt.scatter(df['ECAL_cluster_x_arr'][i],df['ECAL_cluster_y_arr'][i], c='black')
# #     plt.scatter(df['eminus_MCphotondaughters_ECAL_X'][i],df['eminus_MCphotondaughters_ECAL_Y'][i], c='orange')
# #     plt.scatter(df['eminus_ECAL_velotrack_x'][i],df['eminus_ECAL_velotrack_y'][i], c='red', marker='x')
# #     plt.plot([df['eminus_ECAL_x'][i], df['eminus_ECAL_velotrack_x'][i]], [df['eminus_ECAL_y'][i], df['eminus_ECAL_velotrack_y'][i]])
# #     for j in range(10):
# #         if cand_frame['labels'][count]==True:
# #             plt.scatter(cand_frame['x_pos'][count],cand_frame['y_pos'][count], c='blue', marker='x')
# #         count+=1
# #     ax = plt.gca()
# #     ax.set_xlim(-4000,4000)
# #     ax.set_ylim(-4000,4000)
# #     plt.show()
#
#
# #
# #
# # # Get useable values line_y
# # df = get_non_arrays(df_orig)
