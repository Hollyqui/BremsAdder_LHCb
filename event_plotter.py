import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import root_pandas
from itertools import compress

# helper functions
def normalize(x):
    return np.divide(x, np.max(x))

def filter(x, bool_list):
    return list(compress(x, bool_list))

# file import
filename = "/home/felix/PycharmProjects/Thesis/Data/data/DaVinci_PgunElectrons_081119_1000ev.root"
df = root_pandas.read_root(filename)
print(list(df.columns))
df_numpy = np.array(df)

x_arr = []
y_arr = []
e_arr = []
xe_arr = []
ye_arr = []
n_cluster_arr = []
adder_arr = []
for i in range(0, min(len(df), 100)):
    x_arr = []
    y_arr = []
    e_arr = []
    xe_arr = []
    ye_arr = []
    n_cluster_arr = []
    adder_arr = []
    # print(df['ECAL_cluster_x_arr'][i])
    for j in df['ECAL_cluster_x_arr'][i]:
        n_cluster_arr.append(df['N_ECAL_clusters'][i])
        adder_arr.append(df['eminus_HasBremAdded'][i])
        xe_arr.append(df['eminus_ECAL_x'][i])
        ye_arr.append(df['eminus_ECAL_y'][i])
        x_arr.append(j)
    for j in df['ECAL_cluster_y_arr'][i]:
        y_arr.append(j)
    for j in df['ECAL_cluster_e_arr'][i]:
        e_arr.append(j)
    plt.scatter(x_arr, y_arr, c=n_cluster_arr, s=30, marker='o')
    plt.scatter(xe_arr, ye_arr, c=n_cluster_arr, s=30, marker='1')
    axes = plt.gca()
    axes.set_xlim([-4000, 4000])
    axes.set_ylim([-4000, 4000])
    plt.title('Deposits from one electron')
    plt.show()

e_arr = normalize(e_arr)*100



# count_smaller_2 = 0
# count = 0
# for i in range(0, len(df)):
#     if(df['N_ECAL_clusters'][i]<=2):
#         count_smaller_2 +=1
#     count+=1
#
# print(count_smaller_2/count)