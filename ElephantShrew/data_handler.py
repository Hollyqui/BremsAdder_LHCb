from dataloader import Data_loader
from mathematics import Mathematics
import pandas as pd
import numpy as np

#
# filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set"
# df = Data_loader.load(filename, 0, 50)
# df.columns.tolist()



class Data_handler:
    showermax = 12600
    df_orig = None


    def __init__(self, df):
        self.df_orig = df

    """This function returns all values that can be used for predictions so filters out all truth"""
    def return_usable(self, df):
        showermax = 12600
        single_X_train = pd.DataFrame({"eminus_nobrem_P": np.array(df['eminus_nobrem_P'])})
        single_X_train['eminus_nobrem_PT'] = df['eminus_nobrem_PT']
        # single_X_train['eminus_OWNPV_X'] = df['eminus_OWNPV_X']
        # single_X_train['eminus_OWNPV_Y'] = df['eminus_OWNPV_Y']
        # single_X_train['eminus_OWNPV_Z'] = df['eminus_OWNPV_Z']
        # single_X_train['eminus_OWNPV_XERR'] = df['eminus_OWNPV_XERR']
        # single_X_train['eminus_OWNPV_YERR'] = df['eminus_OWNPV_YERR']
        # single_X_train['eminus_OWNPV_ZERR'] = df['eminus_OWNPV_ZERR']
        # single_X_train['eminus_OWNPV_CHI2'] = df['eminus_OWNPV_CHI2']
        # single_X_train['eminus_OWNPV_NDOF'] = df['eminus_OWNPV_NDOF']
        # single_X_train['eminus_IP_OWNPV'] = df['eminus_IP_OWNPV']
        # single_X_train['eminus_IPCHI2_OWNPV'] = df['eminus_IPCHI2_OWNPV']
        # single_X_train['eminus_ORIVX_X'] = df['eminus_ORIVX_X']
        # single_X_train['eminus_ORIVX_Y'] = df['eminus_ORIVX_Y']
        # single_X_train['eminus_ORIVX_Z'] = df['eminus_ORIVX_Z']
        # single_X_train['eminus_ORIVX_XERR'] = df['eminus_ORIVX_XERR']
        # single_X_train['eminus_ORIVX_YERR'] = df['eminus_ORIVX_YERR']
        # single_X_train['eminus_ORIVX_ZERR'] = df['eminus_ORIVX_ZERR']
        # single_X_train['eminus_ORIVX_CHI2'] = df['eminus_ORIVX_CHI2']
        # single_X_train['eminus_ORIVX_NDOF'] = df['eminus_ORIVX_NDOF']
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
        # single_X_train['eminus_PT'] = df['eminus_PT']
        # single_X_train['eminus_PE'] = df['eminus_PE']
        # single_X_train['eminus_PX'] = df['eminus_PX']
        # single_X_train['eminus_PY'] = df['eminus_PY']
        # single_X_train['eminus_PZ'] = df['eminus_PZ']
        # single_X_train['eminus_M'] = df['eminus_M']
        # single_X_train['eminus_ID'] = df['eminus_ID']
        single_X_train['eminus_PIDe'] = df['eminus_PIDe']
        # single_X_train['eminus_PIDmu'] = df['eminus_PIDmu']
        # single_X_train['eminus_PIDK'] = df['eminus_PIDK']
        # single_X_train['eminus_PIDp'] = df['eminus_PIDp']
        # single_X_train['eminus_PIDd'] = df['eminus_PIDd']
        single_X_train['eminus_ProbNNe'] = df['eminus_ProbNNe']
        single_X_train['eminus_ProbNNk'] = df['eminus_ProbNNk']
        single_X_train['eminus_ProbNNp'] = df['eminus_ProbNNp']
        single_X_train['eminus_ProbNNpi'] = df['eminus_ProbNNpi']
        single_X_train['eminus_ProbNNmu'] = df['eminus_ProbNNmu']
        single_X_train['eminus_ProbNNd'] = df['eminus_ProbNNd']
        single_X_train['eminus_ProbNNghost'] = df['eminus_ProbNNghost']
        single_X_train['eminus_hasMuon'] = df['eminus_hasMuon']
        single_X_train['eminus_isMuon'] = df['eminus_isMuon']
        # single_X_train['eminus_hasRich'] = df['eminus_hasRich']
        # single_X_train['eminus_UsedRichAerogel'] = df['eminus_UsedRichAerogel']
        # single_X_train['eminus_UsedRich1Gas'] = df['eminus_UsedRich1Gas']
        # single_X_train['eminus_UsedRich2Gas'] = df['eminus_UsedRich2Gas']
        # single_X_train['eminus_RichAboveElThres'] = df['eminus_RichAboveElThres']
        # single_X_train['eminus_RichAboveMuThres'] = df['eminus_RichAboveMuThres']
        # single_X_train['eminus_RichAbovePiThres'] = df['eminus_RichAbovePiThres']
        # single_X_train['eminus_RichAboveKaThres'] = df['eminus_RichAboveKaThres']
        # single_X_train['eminus_RichAbovePrThres'] = df['eminus_RichAbovePrThres']
        single_X_train['eminus_hasCalo'] = df['eminus_hasCalo']
        # single_X_train['eminus_TRACK_Type'] = df['eminus_TRACK_Type']
        # single_X_train['eminus_TRACK_Key'] = df['eminus_TRACK_Key']
        single_X_train['eminus_TRACK_CHI2NDOF'] = df['eminus_TRACK_CHI2NDOF']
        single_X_train['eminus_TRACK_PCHI2'] = df['eminus_TRACK_PCHI2']
        single_X_train['eminus_TRACK_MatchCHI2'] = df['eminus_TRACK_MatchCHI2']
        single_X_train['eminus_TRACK_GhostProb'] = df['eminus_TRACK_GhostProb']
        single_X_train['eminus_TRACK_CloneDist'] = df['eminus_TRACK_CloneDist']
        single_X_train['eminus_TRACK_Likelihood'] = df['eminus_TRACK_Likelihood']
        # single_X_train['nCandidate'] = df['nCandidate']
        # single_X_train['totCandidates'] = df['totCandidates']
        # single_X_train['EventInSequence'] = df['EventInSequence']
        # single_X_train['N_ECAL_clusters'] = df['N_ECAL_clusters']
        # single_X_train['N_ECAL_photons'] = df['N_ECAL_photons']
        # single_X_train['BCID'] = df['BCID']
        # single_X_train['BCType'] = df['BCType']
        # single_X_train['OdinTCK'] = df['OdinTCK']
        # single_X_train['L0DUTCK'] = df['L0DUTCK']
        # single_X_train['HLT1TCK'] = df['HLT1TCK']
        # single_X_train['HLT2TCK'] = df['HLT2TCK']
        # single_X_train['GpsTime'] = df['GpsTime']
        # single_X_train['Polarity'] = df['Polarity']
        # single_X_train['nPV'] = df['nPV']
        single_X_train['eminus_TRUEID'] = df['eminus_TRUEID']
        single_X_train['eminus_TTstate_x'] = df['eminus_TTstate_x']
        single_X_train['eminus_TTstate_y'] = df['eminus_TTstate_y']
        single_X_train['eminus_TTstate_z'] = df['eminus_TTstate_z']
        single_X_train['eminus_velostate_x'] = df['eminus_velostate_x']
        single_X_train['eminus_velostate_y'] = df['eminus_velostate_y']
        single_X_train['eminus_velostate_z'] = df['eminus_velostate_z']
        tt_dir = np.array([single_X_train['eminus_ECAL_TTtrack_x']-single_X_train['eminus_TTstate_x'],single_X_train['eminus_ECAL_TTtrack_y']-single_X_train['eminus_TTstate_y'],showermax-single_X_train['eminus_TTstate_z']]).T
        velo_dir = np.array([single_X_train['eminus_ECAL_velotrack_x']-single_X_train['eminus_velostate_x'],single_X_train['eminus_ECAL_velotrack_y']-single_X_train['eminus_velostate_y'],showermax-single_X_train['eminus_velostate_z']]).T
        single_X_train['velo_ttrack_angle'] = Mathematics.get_angle(tt_dir, velo_dir)


        # now get arrays
        single_X_train['ECAL_cluster_x_arr'] = df['ECAL_cluster_x_arr']
        single_X_train['ECAL_cluster_y_arr'] = df['ECAL_cluster_y_arr']
        single_X_train['ECAL_cluster_e_arr'] = df['ECAL_cluster_e_arr']


        single_X_train['ECAL_photon_x_arr'] = df['ECAL_photon_x_arr']
        single_X_train['ECAL_photon_y_arr'] = df['ECAL_photon_y_arr']
        single_X_train['ECAL_photon_sprx_arr'] = df['ECAL_photon_sprx_arr']
        single_X_train['ECAL_photon_spry_arr'] = df['ECAL_photon_spry_arr']
        single_X_train['ECAL_photon_CL_arr'] = df['ECAL_photon_CL_arr']
        single_X_train['ECAL_photon_PT_arr'] = df['ECAL_photon_PT_arr']
        return single_X_train

    """Checks whether bremadder would add the particle (that happens if it is the the square spanned by the velo/ttrack projection)"""
    def would_brem_add(self, df,i,j):
        min_x =  min(df['eminus_ECAL_TTtrack_x'][i]-df['eminus_ECAL_TTtrack_sprx'][i],df['eminus_ECAL_velotrack_x'][i]-df['eminus_ECAL_velotrack_sprx'][i])
        max_x = max(df['eminus_ECAL_TTtrack_x'][i]+df['eminus_ECAL_TTtrack_sprx'][i],df['eminus_ECAL_velotrack_x'][i]+df['eminus_ECAL_velotrack_sprx'][i])
        min_y =  min(df['eminus_ECAL_TTtrack_y'][i]-df['eminus_ECAL_TTtrack_spry'][i],df['eminus_ECAL_velotrack_y'][i]-df['eminus_ECAL_velotrack_spry'][i])
        max_y = max(df['eminus_ECAL_TTtrack_y'][i]+df['eminus_ECAL_TTtrack_spry'][i],df['eminus_ECAL_velotrack_y'][i]+df['eminus_ECAL_velotrack_spry'][i])
        if min_x<df['ECAL_cluster_x_arr'][i][j]<max_x:
            if min_y<df['ECAL_cluster_y_arr'][i][j]<max_y:
                return True
        return False

    """Find whether a cluster belongs to an electron daughter"""
    #TODO: make sure it doesn't add the electron cluster itself

    def is_daughter(self, df, df_orig, i, j, max_match=80):
        new_df = df.copy()
        if len(df_orig['eminus_MCphotondaughters_ECAL_X'][i])>0:
            dist = Mathematics.get_distance([new_df['ECAL_cluster_x_arr'][i][j],new_df['ECAL_cluster_y_arr'][i][j]],
                                            [df_orig['eminus_MCphotondaughters_ECAL_X'][i],df_orig['eminus_MCphotondaughters_ECAL_Y'][i]])
            if np.min(dist)<max_match:
                return True
        return False

    """Gets distance between cluster and ttrack/velo projection center"""
    def get_window_dist(self, df, i, j):
        new_df = df.copy()
        projection_x = (new_df['eminus_ECAL_TTtrack_x'][i]+new_df['eminus_ECAL_velotrack_x'][i])/2
        projection_y = (new_df['eminus_ECAL_TTtrack_y'][i]+new_df['eminus_ECAL_velotrack_y'][i])/2
        return np.abs(Mathematics.get_distance([projection_x, projection_y], [new_df['ECAL_cluster_x_arr'][i][j], new_df['ECAL_cluster_y_arr'][i][j]]))


    """Sometimes there are daughter photons next to the electron but the cluster is caused by the electron."""
    """For computational pruposes they are removed in an extra step"""
    def remove_electron_label(self, df):
        new_df = df.copy()
        for i in range(len(new_df)):
            for j in range(len(new_df['ecal_clusters'][i])):
                if new_df['ecal_clusters'][i][j]["is_electron"] == True:
                    new_df['ecal_clusters'][i][j]["label"] = False
        return new_df
    # df.columns.tolist()

    """Takes in a dataframe with dictionaries and adds a field for whether a cluster is an electron"""
    def is_electron(self, df):
        new_df = df.copy()
        for i in range(len(new_df)):
            dist = []
            for j in range(len(new_df['ecal_clusters'][i])):
                dist.append(Mathematics.get_distance([new_df['eminus_ECAL_x'][i], new_df['eminus_ECAL_y'][i]],
                                                [new_df['ecal_clusters'][i][j]["x_pos"],new_df['ecal_clusters'][i][j]["y_pos"]]))
            for j in range(len(new_df['ecal_clusters'][i])):
                if j == np.argmin(dist):
                    new_df['ecal_clusters'][i][j]["is_electron"] = True
                else:
                    new_df['ecal_clusters'][i][j]["is_electron"] = False
        return new_df

    """This function filters a number of candidate photons based on their location"""
    def filter_clusters(self, df, n_candidates=10):
        new_df = df.copy()
        # find the center of ttrack and velotrack
        projection_x = (new_df['eminus_ECAL_TTtrack_x']+new_df['eminus_ECAL_velotrack_x'])/2
        projection_y = (new_df['eminus_ECAL_TTtrack_y']+new_df['eminus_ECAL_velotrack_y'])/2

        # for performance reasons (if it stays as pandas dataframe it is very slow)
        x_arr = np.array(new_df['ECAL_cluster_x_arr'])
        y_arr = np.array(new_df['ECAL_cluster_y_arr'])

        # find the closest cluters to the projection
        for i in range(len(df)):
            dist = Mathematics.get_distance([projection_x[i], projection_y[i]], [x_arr[i],  y_arr[i]])
            # sorts x and y of clusters based on distance
            temp = sorted(zip(dist, x_arr[i], y_arr[i]))
            dist, x, y = map(list, zip(*temp))
            x_arr[i] = x[:10]
            y_arr[i] = y[:10]
        new_df['ECAL_cluster_x_arr'] = x_arr
        new_df['ECAL_cluster_y_arr'] = y_arr
        return new_df
        # df.columns.tolist()

    """For computational speed split the dataframe into subparts"""
    def dictonarize(self, df, matching_region=80):
        new_df = df.copy()
        arr = []
        for i in range(0,len(new_df),500):
            orig = self.df_orig[i:i+500].reset_index().drop(['index'],axis=1)
            arr.append(self.cluster_to_dict(new_df.iloc[i:i+500].reset_index().drop(['index'],axis=1), orig, matching_region))
            print("Progress:", i+500/len(new_df))
        test_dict_usable = arr[0]
        for i in range(1, len(arr)):
            test_dict_usable=test_dict_usable.append(arr[i], ignore_index=True)
        return test_dict_usable

    """The photon positions don't _exactly_ match up with the cluster positions so we find the best match and make each cluster into a dictionary of it's properties"""
    def cluster_to_dict(self, df, df_orig, matching_region = 80):
        new_df = df.copy()
        # cluster prop contains all clusters as dictionaries
        cluster_prop = []
        for i in range(len(new_df)):
            cluster_prop.append([])
            # dict is created for each cluster and the distance to next photon is checked
            for j in range(len(new_df['ECAL_cluster_x_arr'][i])):
                cluster_prop[i].append({"x_pos": new_df['ECAL_cluster_x_arr'][i][j],
                                        "y_pos": new_df['ECAL_cluster_y_arr'][i][j],
                                        "energy": new_df['ECAL_cluster_e_arr'][i][j],
                                        "would_brem_add": self.would_brem_add(new_df, i,j),
                                        "label": self.is_daughter(new_df, df_orig, i, j, max_match=matching_region),
                                        "dist": self.get_window_dist(new_df, i, j)})
                dist = Mathematics.get_distance([new_df['ECAL_cluster_x_arr'][i][j],  new_df['ECAL_cluster_y_arr'][i][j]],
                                                [new_df['ECAL_photon_x_arr'][i], new_df['ECAL_photon_y_arr'][i]])
                # if the distance is below threshold, consider it a photon and add all information about it
                if np.min(dist)<matching_region:
                    cluster_prop[i][j]["isPhot"] = True
                    cluster_prop[i][j]["Phot_sprx"] = new_df['ECAL_photon_sprx_arr'][i][np.argmin(dist)]
                    cluster_prop[i][j]["Phot_spry"] = new_df['ECAL_photon_spry_arr'][i][np.argmin(dist)]
                    cluster_prop[i][j]["Phot_CL"] = new_df['ECAL_photon_CL_arr'][i][np.argmin(dist)]
                    cluster_prop[i][j]["Phot_PT"] = new_df['ECAL_photon_PT_arr'][i][np.argmin(dist)]
                # otherwise just add zeros instead
                else:
                    cluster_prop[i][j]["isPhot"] = False
                    cluster_prop[i][j]["Phot_sprx"] = 0
                    cluster_prop[i][j]["Phot_spry"] = 0
                    cluster_prop[i][j]["Phot_CL"] = 0
                    cluster_prop[i][j]["Phot_PT"] = 0
        new_df['ecal_clusters'] = cluster_prop
        return new_df

    """This method prepares a dataframe that can be used for the classification"""
    """It deletes all truth information and removes arrays/values that cannot be used"""
    def prepare_classification(self, df):
        new_df = df.copy()
        # first drop all arrays except the one containing the dict of cluster properties since
        # they can't be fed to the xgbooster
        new_df = new_df.drop(['ECAL_cluster_x_arr','ECAL_cluster_y_arr','ECAL_cluster_e_arr','ECAL_photon_x_arr','ECAL_photon_y_arr',
                              'ECAL_photon_sprx_arr','ECAL_photon_spry_arr','ECAL_photon_CL_arr','ECAL_photon_PT_arr'], axis=1)

        # now for each photon we make its own row
        rows = []
        for i in range(len(new_df)):
            for j in range(len(new_df['ecal_clusters'][i])):
                rows.append(np.array(list(new_df['ecal_clusters'][i][j].values())))
                rows[-1]=np.concatenate((rows[-1], new_df.iloc[i]))

        # name columns correctly again
        col_labels = np.concatenate((np.array(list(new_df['ecal_clusters'][0][0].keys())),new_df.columns))
        new_df = pd.DataFrame(data=np.array(rows), index=None, columns=col_labels)
        new_df = new_df.drop(['ecal_clusters'], axis=1)
        new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='MC')))]
        new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='TRUE')))]
        return new_df

    """If there is a model to make predictions for whether a cluster is a daughter cluster this"""
    """function can be used to add the information to the dataframe"""
    def assign_prediction(self, dict_df, model, class_frame=None):
        if class_frame is None:
            class_frame = self.prepare_classification(dict_df)
        new_df = dict_df.copy()
        new_class_frame = class_frame.copy()
        x_train = new_class_frame.drop(['label'], axis=1)*1
        preds = model.predict_proba(np.array(x_train)).T[1]
        count = 0
        for i in range(len(new_df)):
            for j in range(len(new_df['ecal_clusters'][i])):
                new_df['ecal_clusters'][i][j]["xgb_pred"] = preds[count]
                count+=1
        return new_df

    """Creates a frame use to train a regressor to predict the true energy"""
    def calc_frame(self, df, n_cand = 3):
        df_orig = self.df_orig
        new_df = df.copy()
        # to remove:
        # new_df = dict_frame.copy()
        # df_orig = df.copy()
        # n_cand= 3


        # add the true moment as a label
        new_df["TRUE_P"] = np.sqrt(df_orig['eminus_TRUEP_X']**2+df_orig['eminus_TRUEP_Y']**2+df_orig['eminus_TRUEP_Z']**2)
        # the following three lines would drop all 0 entries but this is probably not wanted (yet)
        # zero_labels = new_df[new_df['TRUE_P'] == 0 ].index
        # new_df.drop(zero_labels, inplace=True)
        # new_df = new_df.reset_index().drop(['index'], axis=1)

        #sort particles so that high xgb predictions are first (more relevant)

        # for computational reasons make them into np array first
        ecal_clusters = np.array(new_df['ecal_clusters'])
        # get top n_cand most likely clusters and delete rest
        for i in range(len(new_df)):
            ecal_clusters[i] = sorted(ecal_clusters[i], key=lambda k: k['xgb_pred'], reverse=True)[:n_cand]
        new_df['ecal_clusters'] = ecal_clusters


        # put all of the information in one row
        photons = []
        for i in range(len(new_df)):
            photons.append(pd.DataFrame(new_df['ecal_clusters'][i]).values.reshape(-1))
        np.array(photons).shape
        photon_df = pd.DataFrame(photons, index=None, columns=n_cand*pd.DataFrame(new_df['ecal_clusters'][i]).columns.tolist())

        result = pd.concat([photon_df, new_df.drop(['ecal_clusters'], axis=1)], axis=1, sort=False)


        # drop unwanted data
        try:
            result = result.drop(['ECAL_cluster_x_arr','ECAL_cluster_y_arr','ECAL_cluster_e_arr','ECAL_photon_x_arr','ECAL_photon_y_arr',
                                  'ECAL_photon_sprx_arr','ECAL_photon_spry_arr','ECAL_photon_CL_arr','ECAL_photon_PT_arr'], axis=1)
        except:
            pass
        return result

# backup = df.copy()
# dataframe = df.copy()
# import time
# handler = Data_handler(df)
# usable = handler.return_usable(dataframe)
# filtered_usable = handler.filter_clusters(usable)
# dict_usable = handler.cluster_to_dict(filtered_usable)
# electron_usable = handler.is_electron(dict_usable)
# class_frame = handler.prepare_classification(electron_usable)
# class_frame.columns.shape
# electron_usable.columns.shape
# dict_frame = handler.assign_prediction(electron_usable, model, class_frame)
# calc_frame = handler.calc_frame(dict_frame, n_cand=3)
# calc_frame.columns
#
# dict_frame.columns
# class_frame.columns
# electron_usable.columns
#
#
# import xgboost as xgb
#
# model = xgb.XGBClassifier({'nthread':4}) #init model
# model.load_model("C:/Users/felix/Documents/University/Thesis/ElephantShrew/ElePhant_Classifier.model")
#
#
# class_frame
# y_train = class_frame['label']*1
# x_train = class_frame.drop(['label'], axis=1)*1
# preds = model.predict_proba(np.array(x_train)).T[1]
# preds
#
#


























#
# for i in range(1):
#     plt.scatter(handler.df_orig['eminus_MCphotondaughters_ECAL_X'][i],handler.df_orig['eminus_MCphotondaughters_ECAL_Y'][i], marker='3')
# for i in range(10):
#     plt.scatter(class_frame['x_pos'][i], class_frame['y_pos'][i], c='black')
#     plt.scatter(class_frame['x_pos'][i], class_frame['y_pos'][i], c='orange', s=int(class_frame['label'][i]*100))
# class_frame['label'][:10]
#
#
#
#
# class_frame
#
#
# # filtered_usable['ECAL_cluster_y_arr'][0]
# import matplotlib.pyplot as plt
#
# plt.scatter(usable['ECAL_cluster_x_arr'][0],usable['ECAL_cluster_y_arr'][0])
# plt.scatter(df['eminus_ECAL_TTtrack_x'][0], df['eminus_ECAL_TTtrack_y'][0], c='orange')
# plt.scatter(filtered_usable['ECAL_cluster_x_arr'][0],filtered_usable['ECAL_cluster_y_arr'][0], c='red', marker='3')
# plt.scatter(df['eminus_ECAL_x'][0], df['eminus_ECAL_y'][0], c='red')
#
# plt.scatter(usable['ECAL_cluster_x_arr'][0],usable['ECAL_cluster_y_arr'][0])
# for i in range(len(dict_usable['ecal_clusters'][0])):
#     plt.scatter(dict_usable['ecal_clusters'][0][i]["x_pos"],dict_usable['ecal_clusters'][0][i]["y_pos"], s=dict_usable['ecal_clusters'][0][i]["isPhot"]*100, c='orange')
#     plt.scatter(dict_usable['ecal_clusters'][0][i]["x_pos"],dict_usable['ecal_clusters'][0][i]["y_pos"], s=dict_usable['ecal_clusters'][0][i]["is_electron"]*100, c='red')
#
# # plt.scatter(usable['ECAL_photon_x_arr'][0],usable['ECAL_photon_y_arr'][0])
#
#
#
# count = 0
# for i in range(len(dict_usable)):
#     for j in range(len(dict_usable["ecal_clusters"][i])):
#         count+=dict_usable["ecal_clusters"][i][j]["would_brem_add"]
# count
#
#
# count = 0
# for i in range(len(dict_usable)):
#     count+=df["eminus_BremMultiplicity"][i]
# count
#
#
#
# df.columns.tolist()
# tot_count = 0
# for i in range(min(len(df),100)):
#     plt.scatter(df['ECAL_cluster_x_arr'][i],df['ECAL_cluster_y_arr'][i], c='black')
#     min_x =  min(df['eminus_ECAL_TTtrack_x'][i]-df['eminus_ECAL_TTtrack_sprx'][i],df['eminus_ECAL_velotrack_x'][i]-df['eminus_ECAL_velotrack_sprx'][i])
#     max_x = max(df['eminus_ECAL_TTtrack_x'][i]+df['eminus_ECAL_TTtrack_sprx'][i],df['eminus_ECAL_velotrack_x'][i]+df['eminus_ECAL_velotrack_sprx'][i])
#     min_y =  min(df['eminus_ECAL_TTtrack_y'][i]-df['eminus_ECAL_TTtrack_spry'][i],df['eminus_ECAL_velotrack_y'][i]-df['eminus_ECAL_velotrack_spry'][i])
#     max_y = max(df['eminus_ECAL_TTtrack_y'][i]+df['eminus_ECAL_TTtrack_spry'][i],df['eminus_ECAL_velotrack_y'][i]+df['eminus_ECAL_velotrack_spry'][i])
#     plt.scatter(min_x, max_y, color='green', s=10)
#     plt.scatter(min_x, min_y, color='green', s=10)
#     plt.scatter(max_x, min_y, color='green', s=10)
#     plt.scatter(max_x, max_y, color='green', s=10)
#     plt.title(df["eminus_BremMultiplicity"][i])
#     count=0
#     for j in range(len(dict_usable["ecal_clusters"][i])):
#         count+=dict_usable["ecal_clusters"][i][j]["would_brem_add"]
#     print(count)
#     tot_count+=count
#     plt.show()
