import root_pandas
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress

# file import
filename = "/home/felix/PycharmProjects/Thesis/Data/data/full_events_2k.root"
df = root_pandas.read_root(filename)
print(list(df.columns))
df_numpy = np.array(df)

plt.hist(df['eminus_MCphotondaughters_N'], bins='auto')
plt.show()
# helper functions
def normalize(x):
    return np.divide(x, np.max(x))

def filter(x, bool_list):
    return list(compress(x, bool_list))
for i in range(0, min(len(df), 1000)):
    x_arr = []
    y_arr = []
    e_arr = []
    xe_arr = []
    ye_arr = []
    xphoton_arr = []
    yphoton_arr = []
    n_cluster_arr = []
    adder_arr = []
    # print(df['ECAL_cluster_x_arr'][i])
    for j in range(len(df['ECAL_cluster_x_arr'][i])):
        n_cluster_arr.append(df['N_ECAL_clusters'][i])
        adder_arr.append(df['eminus_HasBremAdded'][i])
        xe_arr.append(df['eminus_ECAL_x'][i])
        ye_arr.append(df['eminus_ECAL_y'][i])
        x_arr.append(df['ECAL_cluster_x_arr'][i][j])
        y_arr.append(df['ECAL_cluster_y_arr'][i][j])
    for j in range(len(df['eminus_MCphotondaughters_ECAL_X'][i])):
        xphoton_arr.append(df['eminus_MCphotondaughters_ECAL_X'][i][j])
        yphoton_arr.append(df['eminus_MCphotondaughters_ECAL_Y'][i][j])
    for j in df['ECAL_cluster_e_arr'][i]:
        e_arr.append(j)
    e_arr = normalize(e_arr) * 100
    plt.scatter(x_arr, y_arr, c=n_cluster_arr, s=e_arr, marker='o')
    plt.scatter(xphoton_arr, yphoton_arr, c='orange', s=e_arr, marker='o')
    plt.scatter(xe_arr, ye_arr, c='red', s=e_arr, marker='1')
    axes = plt.gca()
    # axes.set_xlim([-4000, 4000])
    # axes.set_ylim([-4000, 4000])
    n_phot = df['eminus_MCphotondaughters_N'][i]
    title = 'Deposits from one electron, Photons:' + str(n_phot) + '\n N of Electrons added by Bremsadder:' + str(df['eminus_BremMultiplicity'][i])
    plt.title(title)
    plt.show()





# count_smaller_2 = 0
# count = 0
# for i in range(0, len(df)):
#     if(df['N_ECAL_clusters'][i]<=2):
#         count_smaller_2 +=1
#     count+=1
#
# print(count_smaller_2/count)