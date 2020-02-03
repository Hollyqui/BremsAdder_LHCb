import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import root_pandas
from itertools import compress

def normalize(x):
    return np.divide(x, np.max(x))

def filter(x, bool_list):
    return list(compress(x, bool_list))

filename = "/home/felix/PycharmProjects/Thesis/Data/data/DaVinci_PgunElectrons_081119_1000ev.root"
df = root_pandas.read_root('/home/felix/PycharmProjects/Thesis/Data/data/DaVinci_PgunElectrons_081119_1000ev.root')
print(list(df.columns))
df_numpy = np.array(df)
x_arr = []
y_arr = []
e_arr = []
xe_arr = []
ye_arr = []
n_cluster_arr = []
adder_arr = []
for i in range(0, min(len(df), 100000000)):
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


e_arr = normalize(e_arr)*100
fig = plt.figure()
levels = [1, 2, 3, 4, 5, 6]
colors = ['red', 'brown', 'yellow', 'green', 'blue']
cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)

# invert bremsadder list and get all elements that where bremsadder was not active
adder_arr_neg = [not i for i in adder_arr]
# the following line also removes all instances where n_cluster = 1 (i.e. only the mistakes of bremsadder are shown)
for i in range(len(adder_arr)):
    if n_cluster_arr[i] == 1:
        adder_arr_neg[i] = False



x_arr_filtered = filter(x_arr, adder_arr_neg)
y_arr_filtered = filter(y_arr, adder_arr_neg)
e_arr_filtered = filter(e_arr, adder_arr_neg)
xe_arr_filtered = filter(xe_arr, adder_arr_neg)
ye_arr_filtered = filter(ye_arr, adder_arr_neg)
n_cluster_arr_filtered = filter(n_cluster_arr, adder_arr_neg)

plt.scatter(x_arr_filtered, y_arr_filtered, c=n_cluster_arr_filtered, s=e_arr_filtered, cmap=cmap, norm=norm, marker='x')
plt.scatter(xe_arr_filtered, ye_arr_filtered, c=n_cluster_arr_filtered, s=200, cmap=cmap, norm=norm, marker='1', alpha=0.2)

# removes all elements where there is only a single energy deposit, so also only shows bremsadder errors
for i in range(len(adder_arr)):
    if n_cluster_arr[i] >= 2:
        adder_arr[i] = False



x_arr_filtered = filter(x_arr, adder_arr)
y_arr_filtered = filter(y_arr, adder_arr)
e_arr_filtered = filter(e_arr, adder_arr)
xe_arr_filtered = filter(xe_arr, adder_arr)
ye_arr_filtered = filter(ye_arr, adder_arr)
n_cluster_arr_filtered = filter(n_cluster_arr, adder_arr)

plt.scatter(x_arr_filtered, y_arr_filtered, c=n_cluster_arr_filtered, s=e_arr_filtered, cmap=cmap, norm=norm)
plt.scatter(xe_arr_filtered, ye_arr_filtered, c=n_cluster_arr_filtered, s=200, cmap=cmap, norm=norm, marker='2', alpha=0.2)
# plt.scatter(df['eminus_ECAL_x'],df['eminus_ECAL_y'], marker='1', c='black')


plt.colorbar()
# fig.savefig('/home/felix/PycharmProjects/Thesis/Graphs/' + "bremsadder errors colour coded n_clusters x=noadd, o=added" + '.png')
plt.show()
