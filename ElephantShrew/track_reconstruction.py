'''
General TODO: - vectorize everything (takes ca. 20mins/10k rows)
              - clean up so it looks decent
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.misc import comb
showermax = 12650
magnet_start = 2500
magnet_end = 8000






#%%


################### BEZIER CURVE FUNCTIONS

def bernstein_poly(i, n, t):
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    zPoints = np.array([p[2] for p in points])
    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    zvals = np.dot(zPoints, polynomial_array)

    return xvals, yvals, zvals


'''Reconstruct track between velo and ttrack mesurement'''
def reconstruct_velo_tt(df, start, end, smoothness=100, bend_sharpness=1000):
    tracks = []
    for i in range(start, min(len(df),end)):
        # simply creates bezier curve based on velo and tt position as well
        # as the direction
        tt_pos = np.array([df['eminus_TTstate_x'][i], df['eminus_TTstate_y'][i], df['eminus_TTstate_z'][i]])
        tt_proj = np.array([df['eminus_ECAL_TTtrack_x'][i], df['eminus_ECAL_TTtrack_y'][i], showermax])
        tt_proj_unit = tt_proj/np.linalg.norm(tt_proj)
        velo_pos = np.array([df['eminus_velostate_x'][i], df['eminus_velostate_y'][i], df['eminus_velostate_z'][i]])
        velo_proj = np.array([df['eminus_ECAL_velotrack_x'][i], df['eminus_ECAL_velotrack_y'][i], showermax])
        velo_proj_unit = velo_proj/np.linalg.norm(velo_proj)
        e_proj = np.array([df['eminus_ECAL_x'][i], df['eminus_ECAL_y'][i], showermax])
        tt_direction = tt_proj_unit*(-bend_sharpness)+tt_pos
        velo_direction = velo_proj_unit*bend_sharpness+velo_pos
        xvals, yvals, z_vals = bezier_curve([tt_pos, tt_direction, velo_direction, velo_pos], nTimes=smoothness)

        tracks.append([xvals, yvals, z_vals])
    return  np.array(tracks)
# velo_tt_tracks = reconstruct_velo_tt(df, 0, 10, 100, 1000)

# bend_sharpness = 1000
# smoothness = 100
'''reconstructs track between ttrack and ecal measurement'''
def reconstruct_tt_ecal(df, start, end, smoothness=100, bend_sharpness=3500):
    # creates bezier curve based on ttrack position and direction and ecal position
    # of electron
    tracks = []
    for i in range(start, min(len(df),end)):
        tt_pos = np.array([df['eminus_TTstate_x'][i], df['eminus_TTstate_y'][i], df['eminus_TTstate_z'][i]])
        tt_proj = np.array([df['eminus_ECAL_TTtrack_x'][i], df['eminus_ECAL_TTtrack_y'][i], showermax])
        tt_proj_unit = tt_proj/np.linalg.norm(tt_proj)
        tt_direction = tt_proj_unit*(bend_sharpness)+tt_pos
        ecal_pos = np.array([df['eminus_ECAL_x'][i],df['eminus_ECAL_y'][i],showermax])
        xvals, yvals, z_vals = bezier_curve([tt_pos, tt_direction, ecal_pos], nTimes=smoothness)
        tracks.append([xvals, yvals, z_vals])
    return np.array(tracks)


def reconstruct_track(df, start, end, smoothness_velo_tt=100, bend_sharpness_velo_tt=1000, smoothness_magnet=100, bend_sharpness_magnet=3500):
    tracks1 = reconstruct_velo_tt(df, start, end, smoothness=smoothness_velo_tt, bend_sharpness=bend_sharpness_velo_tt)
    tracks2 = reconstruct_tt_ecal(df, start, end, smoothness=smoothness_magnet, bend_sharpness=bend_sharpness_magnet)
    return np.concatenate((tracks2,tracks1), axis=2)

'''projects track reconstruciton on the ecal''' # TODO: Clean up and optimize - terrible function
def project_track(df, track):
    track = track.T
    delta_track = []
    screen_distance = []
    for i in range(len(track)-1):
        delta_track.append(track[i+1]-track[i])
        screen_distance.append(showermax-track[i][2])
    track = track[:-1]
    delta_track = np.array(delta_track)
    screen_distance = np.array(screen_distance)
    projection = []
    for i in range(len(screen_distance)):
        projection.append([screen_distance[i]*(delta_track[i,0]/delta_track[i,2]), screen_distance[i]*(delta_track[i,1]/delta_track[i,2])])
    projection = np.array(projection)
    final_proj = np.array([track[:,0]+projection[:,0],track[:,1]+projection[:,1],[showermax]*len(projection)]).T
    return final_proj

'''Finds the origin of a photon given their track, their projection and ONE clusters'''
def find_origin(track, projections, cluster):
    complete_track = track.T
    proj_x = projections[:,0]
    proj_y = projections[:,1]
    cluster_x = cluster[0]
    cluster_y = cluster[1]
    dist = []
    for x,y in zip(proj_x, proj_y):
        dist.append((x-cluster_x)**2+(y-cluster_y)**2)
    minimum = np.argmin(dist)
    minimum
    complete_track.shape
    origin = complete_track[minimum]
    return origin


# TODO: Change dataframe column to accept reconstruction instead of truth
'''Automatically finds all photon origins based on the position of their
ecal clusters'''
def find_all_origins(df, index, smoothness_velo_tt=10000, smoothness_magnet=10,
                     bend_sharpness_velo_tt=1000, bend_sharpness_magnet=3500):

    # First reconstruct the whole track
    complete_track = reconstruct_track(df, index, index+1,smoothness_velo_tt=smoothness_velo_tt,
                                       smoothness_magnet=smoothness_magnet, bend_sharpness_velo_tt=bend_sharpness_velo_tt,
                                       bend_sharpness_magnet=bend_sharpness_magnet)
    # find projection on ECAL
    projection = project_track(df,complete_track[0])

    # get origins based on track and its projection
    origins = []
    for j in range(len(df['eminus_MCphotondaughters_ECAL_X'][index])):
        origins.append(find_origin(complete_track[0],projection,[df['eminus_MCphotondaughters_ECAL_X'][index][j],df['eminus_MCphotondaughters_ECAL_Y'][index][j]]))

    return np.array(origins)

'''Makes 3D graphs of the track, mctruth photons etc'''
def plot_3D_graphic(df, i):
    # PLOT THE ECAL SCREEN
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(-4000,4000,100)
    y = np.arange(-4000,4000,100)
    Z = np.ones((80,80))*showermax
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X, Y,Z, alpha=0.1, color='blue')
    # Z = np.ones((80,80))*magnet_start
    # ax.plot_surface(X, Y,Z, alpha=0.1, color='yellow')
    # Z = np.ones((80,80))*magnet_end
    # ax.plot_surface(X, Y,Z, alpha=0.1, color='yellow')


    # RECONSTRUCT TRACKS
    complete_track = reconstruct_track(df, i, i+1,smoothness_velo_tt=10000, smoothness_magnet=10, bend_sharpness_velo_tt=1000)
    # feed SINGLE complete track into projection
    projection = project_track(df,complete_track[0])
    # feed SINGLE track, prediction and cluster into find_origin
    origins = []
    first = True
    for j in range(len(df['eminus_MCphotondaughters_ECAL_X'][i])):
        origins.append(find_origin(complete_track[0],projection,[df['eminus_MCphotondaughters_ECAL_X'][i][j],df['eminus_MCphotondaughters_ECAL_Y'][i][j]]))
        if first:
            ax.scatter(origins[j][0], origins[j][1], origins[j][2], c='green', marker='o',s=50, label='Reconstructed Origins')
            first=False
        else:
            ax.scatter(origins[j][0], origins[j][1], origins[j][2], c='green', marker='o',s=50)

    # PLOT ALL OTHER INFORMATION WE WANT TO INVESTIGATE
    first=True
    for j in range(len(df['eminus_MCphotondaughters_ECAL_X'][i])):
        if first:
            ax.plot([df['eminus_MCphotondaughters_orivx_X'][i][j],df['eminus_MCphotondaughters_ECAL_X'][i][j]],[df['eminus_MCphotondaughters_orivx_Y'][i][j],df['eminus_MCphotondaughters_ECAL_Y'][i][j]],[df['eminus_MCphotondaughters_orivx_Z'][i][j],showermax], c='orange', label='photon trajectory')
            first=False
        else:
            ax.plot([df['eminus_MCphotondaughters_orivx_X'][i][j],df['eminus_MCphotondaughters_ECAL_X'][i][j]],[df['eminus_MCphotondaughters_orivx_Y'][i][j],df['eminus_MCphotondaughters_ECAL_Y'][i][j]],[df['eminus_MCphotondaughters_orivx_Z'][i][j],showermax], c='orange')

    #     ax.scatter(df['eminus_MCphotondaughters_orivx_X'][i][j],df['eminus_MCphotondaughters_orivx_Y'][i][j],df['eminus_MCphotondaughters_orivx_Z'][i][j], c='orange', marker='o', s=50, label='True Origins')
    #     ax.scatter(df['eminus_MCphotondaughters_ECAL_X'][i][j],df['eminus_MCphotondaughters_ECAL_Y'][i][j],showermax, c='orange', label='Photon Cluster')
    # ax.plot([df['eminus_MCphotondaughters_orivx_X'][i],df['eminus_MCphotondaughters_ECAL_X'][i]],[df['eminus_MCphotondaughters_orivx_Y'][i],df['eminus_MCphotondaughters_ECAL_Y'][i]],[df['eminus_MCphotondaughters_orivx_Z'][i],showermax], c='orange', label='photon trajectory')
    ax.scatter(df['eminus_MCphotondaughters_orivx_X'][i],df['eminus_MCphotondaughters_orivx_Y'][i],df['eminus_MCphotondaughters_orivx_Z'][i], c='orange', marker='o', s=50, label='True Origins')
    ax.scatter(df['eminus_MCphotondaughters_ECAL_X'][i],df['eminus_MCphotondaughters_ECAL_Y'][i],showermax, c='orange', label='Photon Cluster')

    ax.scatter(df['ECAL_cluster_x_arr'][i], df['ECAL_cluster_y_arr'][i], zs=showermax, zdir='z', c="black", label='ECAL Clusters')
    # ax.scatter(df['eminus_TTstate_x'][i], df['eminus_TTstate_y'][i], df['eminus_TTstate_z'][i], c='red')
    # ax.scatter(df['eminus_velostate_x'][i],df['eminus_velostate_y'][i],df['eminus_velostate_z'][i], c='blue')
    ax.scatter(df['eminus_ECAL_x'][i], df['eminus_ECAL_y'][i],showermax, c='blue', label='Electron Cluster')

    # ax.plot([df['eminus_TTstate_x'][i],df['eminus_ECAL_TTtrack_x'][i]], [df['eminus_TTstate_y'][i],df['eminus_ECAL_TTtrack_y'][i]], [df['eminus_TTstate_z'][i],showermax], c='red')
    # ax.plot([df['eminus_velostate_x'][i],df['eminus_ECAL_velotrack_x'][i]],[df['eminus_velostate_y'][i],df['eminus_ECAL_velotrack_y'][i]],[df['eminus_velostate_z'][i],showermax], c='blue')


    ax.plot(complete_track[0][0], complete_track[0][1], complete_track[0][2], c='blue', label='Electron Track')
    ax.plot(projection[:,0],projection[:,1],projection[:,2], c='yellow', label='Path Projection')


    # set plot dimensions
    ax.set_ylabel("Y displacement")
    ax.set_xlabel("X displacement")
    ax.set_zlabel("Distance")
    ax.set_xlim(-4500,4500)
    ax.set_ylim(-4500,4500)
    ax.set_zlim(0,13000)
    plt.title(str(df['eminus_MCphotondaughters_N'][i]) + " Daughter Photon(s)")
    plt.legend()
    plt.show()


#%%
############################ PLOTTING AREA ##################################
# # # READ DATAFRAME
# try:
#     filename = "C:/Users/felix/Documents/University/Thesis/track_electron_set"
#     df = pd.read_pickle(filename)
# except:
#     filename = "C:/Users/szymo/Documents/track_electron_set"
#     df = pd.read_pickle(filename)
# START_ROW = 0
# MAX_ROWS = 20000
#
# df = df[START_ROW:MAX_ROWS+START_ROW]
# len(df)
# df = df.reset_index()
# pd.set_option('display.max_columns', 1000)
# print(df.columns.tolist())
#
# #
# # origins = []
# # time1 = time.perf_counter()
# # for i in range(len(df)):
# #     origins.append(find_all_origins(df, i))
# # time2 = time.perf_counter()
# # time2-time1
# # origins[0]
#
#
# n_plots = 1
# start_plot = 1
# for i in range(start_plot, start_plot+n_plots):
#     plot_3D_graphic(df, i)
