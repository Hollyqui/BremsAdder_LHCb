import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from itertools import compress

#%%

############ SET GLOBAL VARIABLES ###################

N_PLOTS = 100 # max number of plots
MAX_ROWS = 10000 # max numbers of DF rows read
DAUGHTER_CLUSTER_MATCH_MAX = 75 # max distance at which a cluster is matched to a photon daughter
#load data
filename = "C:/Users/felix/Documents/University/Thesis/final_large_electron_set"
df = pd.read_pickle(filename)
df = df[:MAX_ROWS]
pd.set_option('display.max_columns', 1000)



origin = [0,0]
ex = df['eminus_ECAL_x']
ey = df['eminus_ECAL_y']
clusters_x = df['ECAL_cluster_x_arr']
clusters_y = df['ECAL_cluster_y_arr']
daughters_x = df['eminus_MCphotondaughters_ECAL_X']
daughters_y = df['eminus_MCphotondaughters_ECAL_Y']
photon_clusters_x = df['ECAL_photon_x_arr']
photon_clusters_y = df['ECAL_photon_y_arr']
velo_x = df['eminus_ECAL_velotrack_x']
velo_y = df['eminus_ECAL_velotrack_y']
ttrack_x = df['eminus_ECAL_TTtrack_x']
ttrack_y = df['eminus_ECAL_TTtrack_y']
line_x, line_y = (velo_x+ttrack_x)/2, (velo_y+ttrack_y)/2

#%%

############# HELPER FUNCTIONS ######################
def arr_filter(x, bool_list):
    return list(compress(x, bool_list))
def get_distance(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)


def center_values(center, data_points):
    return np.subtract(center,data_points)

def rotate(line_end, data):
    line_x, line_y = line_end
    x, y = data
    radians = -np.arcsin(line_y/get_distance(origin, [line_x, line_y]))
    xx = x * np.cos(radians) + y * np.sin(radians)
    yy = (y/get_distance(origin, [x, y])+np.sin(radians))*get_distance(origin, [x, y])
    return xx, yy

def move_photons(photons, clusters):
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
def norm(line_end, data):
    # normalized distances
    normalization = (1/(line_end[0]-origin[0]))*2000
    return data[0]*normalization, data[1]*normalization

def get_scaling(line_end):
    return np.abs((1/(line_end[0]-origin[0]))*2000)

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

def get_candidates():
    for i in range(len(df)):
        p1 = np.array(origin)
        p2 = np.array([norm_line_x[i],norm_line_y[i]])
        p3 = []
        for x,y in zip(norm_centered_clusters_with_daughter_x[i], norm_centered_clusters_with_daughter_y[i]):
            p3.append([x,y])
        p3 = np.array(p3)
        norm_dist_to_line.append(abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)))

def find_daughter_clusters(daughters, clusters, scaling, electron_cluster_x):
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


#%%

center = [ex, ey]

# center datapoints around electron
centered_clusters_x, centered_clusters_y = center_values(center, [clusters_x, clusters_y])
centered_daughters_x, centered_daughters_y = center_values(center, [daughters_x, daughters_y])
centered_velo_x, centered_velo_y = center_values(center,[velo_x, velo_y])
centered_ttrack_x, centered_ttrack_y = center_values(center, [ttrack_x, ttrack_y])
centered_line_x, centered_line_y = center_values(center, [line_x, line_y])
centered_photon_clusters_x, centered_photon_clusters_y = center_values(center, [photon_clusters_x, photon_clusters_y])


# rotate
line_end = [centered_line_x, centered_line_y]
rot_cen_clusters_x, rot_cen_clusters_y = copy.deepcopy(centered_clusters_x), copy.deepcopy(centered_clusters_y)
rot_cen_daughters_x, rot_cen_daughters_y = copy.deepcopy(centered_daughters_x), copy.deepcopy(centered_clusters_y)
rot_cen_photon_clusters_x, rot_cen_photon_clusters_y = copy.deepcopy(centered_photon_clusters_x),copy.deepcopy(centered_photon_clusters_y)

for i in range(len(centered_clusters_x)):
    rot_cen_clusters_x[i], rot_cen_clusters_y[i] = rotate([centered_line_x[i], centered_line_y[i]], [centered_clusters_x[i], centered_clusters_y[i]])
    rot_cen_daughters_x[i], rot_cen_daughters_y[i] = rotate([centered_line_x[i], centered_line_y[i]],[centered_daughters_x[i], centered_daughters_y[i]])
    rot_cen_photon_clusters_x[i], rot_cen_photon_clusters_y[i] = rotate([centered_line_x[i], centered_line_y[i]], [centered_photon_clusters_x[i], centered_photon_clusters_y[i]])
rot_cen_line_x, rot_cen_line_y = rotate([centered_line_x, centered_line_y],[centered_line_x, centered_line_y])

plot_graph(clusters_x=rot_cen_clusters_x, clusters_y=rot_cen_clusters_y, photon_clusters_x=rot_cen_photon_clusters_x, photon_clusters_y=rot_cen_photon_clusters_y)


# normalize
for i in range(len(centered_clusters_x)):
    rot_cen_clusters_x[i], rot_cen_clusters_y[i] = norm([rot_cen_line_x[i],rot_cen_line_y[i]], [rot_cen_clusters_x[i], rot_cen_clusters_y[i]])
    rot_cen_daughters_x[i], rot_cen_daughters_y[i] = norm([rot_cen_line_x[i],rot_cen_line_y[i]],[rot_cen_daughters_x[i], rot_cen_daughters_y[i]])
    rot_cen_photon_clusters_x[i], rot_cen_photon_clusters_y[i] = norm([centered_line_x[i], centered_line_y[i]], [rot_cen_photon_clusters_x[i], rot_cen_photon_clusters_y[i]])
rot_cen_line_x, rot_cen_line_y = norm([rot_cen_line_x,rot_cen_line_y],[rot_cen_line_x, rot_cen_line_y])

plot_graph(clusters_x=rot_cen_clusters_x, clusters_y=rot_cen_clusters_y, photon_clusters_x=rot_cen_photon_clusters_x, photon_clusters_y=rot_cen_photon_clusters_y)


# find electron clusters based on mc truth information

e_cluster_x, e_cluster_y = find_electron([0,0], [rot_cen_clusters_x, rot_cen_clusters_y])
scaling_arr = get_scaling(line_end)

daughter_clusters_x, daughter_clusters_y = find_daughter_clusters([rot_cen_daughters_x, rot_cen_daughters_y], [rot_cen_clusters_x, rot_cen_clusters_y], scaling_arr, e_cluster_x)

plot_graph(overlay=True, clusters_x=rot_cen_clusters_x, clusters_y=rot_cen_clusters_y,
            daughters_x=rot_cen_daughters_x, daughters_y=rot_cen_daughters_y,
            line_x=rot_cen_line_x, line_y=rot_cen_line_y, electron_cluster_x=e_cluster_x,
            electron_cluster_y=e_cluster_y, daughter_clusters_x=daughter_clusters_x,
            daughter_clusters_y=daughter_clusters_y, photon_clusters_x=rot_cen_photon_clusters_x,
            photon_clusters_y=rot_cen_photon_clusters_y)

#%%
# create histogram of how many daughter clusters there are
len(daughter_clusters_x[30])
histogram = []
for i in range(len(daughter_clusters_x)):
    histogram.append((len(daughter_clusters_x[i])))

plt.hist(df['eminus_MCphotondaughters_N'])
plt.hist(histogram, bins=11, range=(0,11))
plt.hist(df['eminus_MCphotondaughters_N']-histogram, bins=11,range=(0,11))

#%%


# plt.scatter(df['ECAL_cluster_x_arr'][0], df['ECAL_cluster_y_arr'][0], c='blue')
# plt.scatter(df['ECAL_photon_x_arr'][0],df['ECAL_photon_y_arr'][0], c='orange', marker='x')
# plt.show()
# df['ECAL_photon_x_arr'][0][0] in df['ECAL_cluster_x_arr'][0]




# distances_squared = norm_centered_clusters_with_daughter_x**2+norm_centered_clusters_with_daughter_y**2
# norm_dist_to_line = []
#
#     # for j in range(len(norm_dist_to_line[i])-1):
#     #     norm_dist_to_line[i][j] += 100*(np.sqrt(distances_squared[i][j])+np.sqrt((norm_line_x[i]-norm_centered_clusters_with_daughter_x[i][j])**2+(norm_line_y[i]-norm_centered_clusters_with_daughter_y[i][j])**2))/(np.sqrt(norm_line_x[i]**2+norm_line_y[i]**2))
# norm_dist_to_line = np.array(norm_dist_to_line)
# # norm_dist_to_line[1]
# size = (1/norm_dist_to_line)*100
