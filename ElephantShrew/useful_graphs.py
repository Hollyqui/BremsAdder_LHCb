# from dataloader import Data_loader
# from mathematics import Mathematics
import pandas as pd
import numpy as np
from data_handler import Data_handler
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as patches

filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set_down"
# test_df = Data_loader.load(filename, 100000, 100200)
# test_df.to_pickle("plotting_orig_df")
test_df = pd.read_pickle("plotting_orig_df")
test_handler = Data_handler(test_df)
test_df.columns.tolist()

test_df.columns.tolist()

for i in range(10):
    fig, ax = plt.subplots()
    ax.scatter(test_df['ECAL_cluster_x_arr'][i], test_df['ECAL_cluster_y_arr'][i], c='black', label='Clusters')
    ax.scatter(test_df['eminus_MCphotondaughters_ECAL_X'][i], test_df['eminus_MCphotondaughters_ECAL_Y'][i], c='orange', label='Photon Daughters')

    min_x =  min(test_df['eminus_ECAL_TTtrack_x'][i]-test_df['eminus_ECAL_TTtrack_sprx'][i],test_df['eminus_ECAL_velotrack_x'][i]-test_df['eminus_ECAL_velotrack_sprx'][i])
    max_x = max(test_df['eminus_ECAL_TTtrack_x'][i]+test_df['eminus_ECAL_TTtrack_sprx'][i],test_df['eminus_ECAL_velotrack_x'][i]+test_df['eminus_ECAL_velotrack_sprx'][i])
    min_y =  min(test_df['eminus_ECAL_TTtrack_y'][i]-test_df['eminus_ECAL_TTtrack_spry'][i],test_df['eminus_ECAL_velotrack_y'][i]-test_df['eminus_ECAL_velotrack_spry'][i])
    max_y = max(test_df['eminus_ECAL_TTtrack_y'][i]+test_df['eminus_ECAL_TTtrack_spry'][i],test_df['eminus_ECAL_velotrack_y'][i]+test_df['eminus_ECAL_velotrack_spry'][i])
    ax.add_patch(
        patches.Rectangle(
            xy=(min_x, min_y),  # point of origin.
            width=abs(min_x-max_x),
            height=abs(min_y-max_y),
            linewidth=1,
            color='red',
            fill=False
            )
    )
    plt.title("All clusters and Photon Daughters")
    ax.legend()
    plt.show()

# abs(test_df['eminus_ECAL_TTtrack_x'][i]-test_df['eminus_ECAL_velotrack_x'][i])
# abs(test_df['eminus_ECAL_TTtrack_y'][i]-test_df['eminus_ECAL_velotrack_y'][i]),
#
# for i in range(10):
#     fig, ax = plt.subplots()
#     ax.scatter(test_df['ECAL_cluster_x_arr'][i], test_df['ECAL_cluster_y_arr'][i], c='black', label='Clusters', s=100)
#     ax.scatter(test_df['ECAL_photon_x_arr'][i], test_df['ECAL_photon_y_arr'][i], c='orange', label='Photons', s=100, alpha=0.7)
#     plt.title("Clusters and Photons")
#     ax.legend()
#     plt.show()





# for i in range(1000):
#     print("Problem:",i)
#     print("Car A:")
#     print("starts at "+str(np.random.randint(0,20))+" km")
#     print("accelerates at "+str(np.random.randint(0,20)/10)+" m/s²")
#     print("starts at" +str(np.random.randint(0,40))+" m/s")
#
#     print("Car B:")
#     print("starts at "+str(np.random.randint(0,20))+" km")
#     print("doesn't accelerate")
#     print("starts at " +str(np.random.randint(0,40))+" m/s")
#     print("starts " +str(np.random.randint(0,50))+" seconds late")
