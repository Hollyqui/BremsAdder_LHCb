import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#load data
filename = "C:/Users/felix/Documents/University/Thesis/final_small_electron_set"
df = pd.read_pickle(filename)
pd.set_option('display.max_columns', 1000)
df

# in case we don't want to use the whole df
MAX_ROWS = 1000
df = df[:MAX_ROWS]
#%%
# combine photon daughter with other clusters
ex = df['eminus_ECAL_x']
ey = df['eminus_ECAL_y']
clusters_with_daughter_x = []
clusters_with_daughter_y = []
clusters_x = df['ECAL_cluster_x_arr']
clusters_y = df['ECAL_cluster_y_arr']
daughters_x = df['eminus_MCphotondaughters_ECAL_X']
daughters_y = df['eminus_MCphotondaughters_ECAL_Y']
velo_x = df['eminus_ECAL_velotrack_x']
velo_y = df['eminus_ECAL_velotrack_y']
ttrack_x = df['eminus_ECAL_TTtrack_x']
ttrack_y = df['eminus_ECAL_TTtrack_y']

# comine clusters and daughters to np array
for i in range(len(df)):
    clusters_with_daughter_x.append([])
    clusters_with_daughter_y.append([])
    clusters_with_daughter_x[i] = np.concatenate((clusters_x[i],daughters_x[i]))
    clusters_with_daughter_y[i] = np.concatenate((clusters_y[i],daughters_y[i]))
clusters_with_daughter_x = np.array(clusters_with_daughter_x)
clusters_with_daughter_y = np.array(clusters_with_daughter_y)


# just graph some images
test_length = 10
for i in range(min(len(df), test_length)):
    plt.scatter(clusters_with_daughter_x[i], clusters_with_daughter_y[i], c='black', s=30, marker='o')
    plt.scatter(daughters_x[i], daughters_y[i], c='orange',s=60, marker='3')
    plt.scatter(ex[i], ey[i], c='blue',s=60, marker='3')
    axes = plt.gca()
    axes.set_xlim([-4000, 4000])
    axes.set_ylim([-4000, 4000])
    plt.show()
#%%

# center all data around electron
centered_clusters_with_daughter_x = np.subtract(ex,clusters_with_daughter_x)
centered_clusters_with_daughter_y = np.subtract(ey,clusters_with_daughter_y)
centered_daughters_x = np.subtract(ex,daughters_x)
centered_daughters_y = np.subtract(ey,daughters_y)
centered_velo_x = np.subtract(ex,velo_x)
centered_velo_y = np.subtract(ey,velo_y)
centered_ttrack_x = np.subtract(ex,ttrack_x)
centered_ttrack_y = np.subtract(ey,ttrack_y)


# now find closest cluster to electron (center) and assign it to electron
distances_squared = centered_clusters_with_daughter_x**2+centered_clusters_with_daughter_y**2
electron_cluster_x = np.ndarray(len(distances_squared))
electron_cluster_y = np.ndarray(len(distances_squared))
for i in range(len(distances_squared)):
    minimum = np.argmin(distances_squared[i])
    electron_cluster_x[i] = centered_clusters_with_daughter_x[i][minimum]
    electron_cluster_y[i] = centered_clusters_with_daughter_y[i][minimum]


# now check distance of every point to the blue line. Here p1 is the origin,
# p2 is the average between velo track and ttrack and p3 iterates through all
# clusters in clusters_with_daughter
dist_to_line = []
for i in range(len(df)):
    dist_to_line.append([])
    p1 = np.array([0,0])
    p2 = np.array([(centered_ttrack_x[i]-centered_velo_x[i])/2,(centered_ttrack_y[i]-centered_velo_y[i])/2])
    p3 = np.array([clusters_with_daughter_x[i], clusters_with_daughter_y[i]]).T
    dist_to_line[-1] = np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)


# just to make sure everything is working:
for i in range(min(len(df), test_length)):
    plt.scatter(centered_clusters_with_daughter_x[i], centered_clusters_with_daughter_y[i], c='black', s=30, marker='o')
    plt.scatter(electron_cluster_x[i], electron_cluster_y[i], c='green',s=50, marker='o')
    plt.scatter(centered_daughters_x[i], centered_daughters_y[i], c='orange',s=60, marker='3')
    plt.plot([0, centered_ttrack_x[i]], [0, centered_ttrack_y[i]])
    plt.plot([0, centered_velo_x[i]],[0, centered_velo_y[i]])
    axes = plt.gca()
    axes.set_xlim([-4000, 4000])
    axes.set_ylim([-4000, 4000])
    plt.show()
