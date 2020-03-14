import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from itertools import compress
#load data
filename = "C:/Users/felix/Documents/University/Thesis/final_small_electron_set"
df = pd.read_pickle(filename)
pd.set_option('display.max_columns', 1000)
# df


#%%

############################# HELPER FUNCTIONS ############################
origin = [0,0]

def arr_filter(x, bool_list):
    return list(compress(x, bool_list))
def get_distance(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def rotate(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * np.cos(radians) + y * np.sin(radians)
    yy = (y/get_distance(origin, [x, y])+np.sin(radians))*get_distance(origin, [x, y])
    return xx, yy


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
# for i in range(min(len(df), test_length)):
#     plt.scatter(clusters_with_daughter_x[i], clusters_with_daughter_y[i], c='black', s=30, marker='o')
#     plt.scatter(daughters_x[i], daughters_y[i], c='orange',s=60, marker='3')
#     plt.scatter(ex[i], ey[i], c='blue',s=60, marker='3')
#     axes = plt.gca()
#     axes.set_xlim([-4000, 4000])
#     axes.set_ylim([-4000, 4000])
#     plt.show()
#%%

# center all data around electron
orig_centered_clusters_with_daughter_x = np.subtract(ex,clusters_with_daughter_x)
orig_centered_clusters_with_daughter_y = np.subtract(ey,clusters_with_daughter_y)
orig_centered_daughters_x = np.subtract(ex,daughters_x)
orig_centered_daughters_y = np.subtract(ey,daughters_y)
orig_centered_velo_x = np.subtract(ex,velo_x)
orig_centered_velo_y = np.subtract(ey,velo_y)
orig_centered_ttrack_x = np.subtract(ex,ttrack_x)
orig_centered_ttrack_y = np.subtract(ey,ttrack_y)
centered_clusters_with_daughter_x = copy.deepcopy(orig_centered_clusters_with_daughter_x)
centered_clusters_with_daughter_y = copy.deepcopy(orig_centered_clusters_with_daughter_y)
centered_daughters_x = copy.deepcopy(orig_centered_daughters_x)
centered_daughters_y = copy.deepcopy(orig_centered_daughters_y)
# now find closest cluster to electron (center) and assign it to electron
distances_squared = orig_centered_clusters_with_daughter_x**2+orig_centered_clusters_with_daughter_y**2
electron_cluster_x = np.ndarray(len(distances_squared))
electron_cluster_y = np.ndarray(len(distances_squared))
for i in range(len(distances_squared)):
    minimum = np.argmin(distances_squared[i])
    electron_cluster_x[i] = orig_centered_clusters_with_daughter_x[i][minimum]
    electron_cluster_y[i] = orig_centered_clusters_with_daughter_y[i][minimum]
orig_line_x = (orig_centered_ttrack_x+orig_centered_velo_x)/2
orig_line_y = (orig_centered_ttrack_y+orig_centered_velo_y)/2


# just to make sure everything is working:
test_length= 0
for i in range(min(len(df), test_length)):
    print(i)
    # plt.scatter(centered_clusters_with_daughter_x[i], centered_clusters_with_daughter_y[i], c=cm.rainbow(size[i]), s=30, marker='o')
    plt.scatter(orig_centered_clusters_with_daughter_x[i], orig_centered_clusters_with_daughter_y[i], c='black', s=30, marker='o')
    plt.scatter(orig_centered_daughters_x[i], orig_centered_daughters_y[i], c='orange',s=60, marker='o')
    # plt.scatter(filtered_ctr_clst_daught_x[i], filtered_ctr_clst_daught_y[i], c='blue', s=30, marker='o')
    plt.scatter(electron_cluster_x[i], electron_cluster_y[i], c='yellow',s=50, marker='x')
    plt.plot([0, orig_line_x[i]], [0, orig_line_y[i]])
    axes = plt.gca()
    axes.set_xlim([-4000, 4000])
    axes.set_ylim([-4000, 4000])
    plt.show()

# rotate all points so line is horizontal
alpha = np.arcsin(orig_line_y/get_distance(origin, [orig_line_x, orig_line_y]))
line_x, line_y = rotate([orig_line_x, orig_line_y], -alpha)
centered_velo_x, centered_velo_y = rotate([orig_centered_velo_x,orig_centered_velo_y], -alpha)
centered_ttrack_x, centered_ttrack_y = rotate([orig_centered_ttrack_x, orig_centered_ttrack_y], -alpha)
# orig_centered_clusters_with_daughter_x[i]
for i in range(len(df)):
    centered_clusters_with_daughter_x[i], centered_clusters_with_daughter_y[i] = rotate([orig_centered_clusters_with_daughter_x[i], orig_centered_clusters_with_daughter_y[i]], -alpha[i])
    centered_daughters_x[i], centered_daughters_y[i] = rotate([centered_daughters_x[i], centered_daughters_y[i]], -alpha[i])

# normalized distances
normalization = (1/line_x)*2000
norm_centered_clusters_with_daughter_x, norm_centered_clusters_with_daughter_y = centered_clusters_with_daughter_x*normalization, centered_clusters_with_daughter_y*normalization
norm_centered_daughters_x, norm_centered_daughters_y = np.array(centered_daughters_x)*normalization, np.array(centered_daughters_y)*normalization
# np.array(filtered_ctr_clst_daught_x)
# norm_filtered_ctr_clst_daught_x, norm_filtered_ctr_clst_daught_y = np.array(filtered_ctr_clst_daught_x)*normalization, np.array(filtered_ctr_clst_daught_y)*normalization
norm_electron_cluster_x, norm_electron_cluster_y = electron_cluster_x*normalization,electron_cluster_y*normalization
norm_line_x, norm_line_y = line_x*normalization, line_y*normalization



test_length = 100
for i in range(min(len(df), test_length)):
    print(i)
    plt.scatter(norm_centered_clusters_with_daughter_x[i], norm_centered_clusters_with_daughter_y[i], c='black', s=30, marker='o')
    # plt.scatter(centered_clusters_with_daughter_x[i], centered_clusters_with_daughter_y[i], c='black', s=30, marker='o')
    plt.scatter(norm_centered_daughters_x[i], norm_centered_daughters_y[i], c='orange',s=60, marker='o')
    # plt.scatter(filtered_ctr_clst_daught_x[i], filtered_ctr_clst_daught_y[i], c='blue', s=30, marker='1')
    plt.scatter(norm_electron_cluster_x[i], norm_electron_cluster_y[i], c='yellow',s=50, marker='o')
    plt.title(df['eminus_HasBremAdded'][i])
    plt.plot([0, norm_line_x[i]], [0, norm_line_y[i]])
axes = plt.gca()
axes.set_xlim([-4000, 4000])
axes.set_ylim([-4000, 4000])
plt.show()


# now check distance of every point to the blue line. Here p1 is the origin,
# p2 is the average between velo track and ttrack and p3 iterates through all
# clusters in clusters_with_daughter. TODO: Vectorize this for efficiency
distances_squared = norm_centered_clusters_with_daughter_x**2+norm_centered_clusters_with_daughter_y**2
norm_dist_to_line = []
for i in range(len(df)):
    p1 = np.array(origin)
    p2 = np.array([norm_line_x[i],norm_line_y[i]])
    p3 = []
    for x,y in zip(norm_centered_clusters_with_daughter_x[i], norm_centered_clusters_with_daughter_y[i]):
        p3.append([x,y])
    p3 = np.array(p3)
    norm_dist_to_line.append(abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)))
    for j in range(len(norm_dist_to_line[i])-1):
        norm_dist_to_line[i][j] += 100*(np.sqrt(distances_squared[i][j])+np.sqrt((norm_line_x[i]-norm_centered_clusters_with_daughter_x[i][j])**2+(norm_line_y[i]-norm_centered_clusters_with_daughter_y[i][j])**2))/(np.sqrt(norm_line_x[i]**2+norm_line_y[i]**2))
norm_dist_to_line = np.array(norm_dist_to_line)
# norm_dist_to_line[1]
size = (1/norm_dist_to_line)*100

max_dist = 200.0
filter_array = copy.deepcopy(norm_dist_to_line)
filter_array[0][0] = 0
for i in range(len(filter_array)):
    for j in range(len(filter_array[i])):
        if filter_array[i][j] >= max_dist:
            filter_array[i][j] = False
        else:
            filter_array[i][j] = True
# filter_array[0]
filtered_ctr_clst_daught_x = []
filtered_ctr_clst_daught_y = []

for i in range(len(centered_clusters_with_daughter_x)):
    filtered_ctr_clst_daught_x.append(arr_filter(norm_centered_clusters_with_daughter_x[i], filter_array[i]))
    filtered_ctr_clst_daught_y.append(arr_filter(norm_centered_clusters_with_daughter_y[i], filter_array[i]))
filtered_ctr_clst_daught_x = np.array(filtered_ctr_clst_daught_x)
filtered_ctr_clst_daught_y = np.array(filtered_ctr_clst_daught_y)

filtered_ctr_clst_daught_x[0]


test_length = 100
for i in range(min(len(df), test_length)):
    print(i)
    plt.scatter(norm_centered_clusters_with_daughter_x[i], norm_centered_clusters_with_daughter_y[i], c=cm.rainbow(size[i]), s=30, marker='o')
    # plt.scatter(centered_clusters_with_daughter_x[i], centered_clusters_with_daughter_y[i], c='black', s=30, marker='o')
    plt.scatter(norm_centered_daughters_x[i], norm_centered_daughters_y[i], c='orange',s=60, marker='o')
    plt.scatter(filtered_ctr_clst_daught_x[i], filtered_ctr_clst_daught_y[i], c='blue', s=30, marker='1')
    plt.scatter(norm_electron_cluster_x[i], norm_electron_cluster_y[i], c='yellow',s=50, marker='o')
    plt.title(df['eminus_HasBremAdded'][i])
    plt.plot([0, norm_line_x[i]], [0, norm_line_y[i]])
    axes = plt.gca()
    axes.set_xlim([-4000, 4000])
    axes.set_ylim([-4000, 4000])
    plt.show()
