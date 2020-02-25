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

# this method of processing the data is a bit ugly but works - consider this implementation preliminary/subjects to
# change
x_arr = []
y_arr = []
e_arr = []
xe_arr = []
ye_arr = []
n_cluster_arr = []
adder_arr = []
for i in range(0, min(len(df), 1000000000)):
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

# normalized and creates colourmap
e_arr = normalize(e_arr)*100
fig = plt.figure()
levels = [1, 2, 3, 4, 5, 6]
colors = ['red', 'brown', 'orange', 'green', 'blue']
cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)


'''Plots the ecal values and electron positions'''
# invert bremsadder list and get all elements that where bremsadder was not active
adder_arr_inverse = [not i for i in adder_arr]

# the following line also removes all instances where n_cluster = 1 (i.e. only the mistakes of bremsadder are shown)
# comment for loop if you want to see all events
for i in range(len(adder_arr)):
    if n_cluster_arr[i] == 1:
        adder_arr_inverse[i] = False



x_arr_filtered = filter(x_arr, adder_arr_inverse)
y_arr_filtered = filter(y_arr, adder_arr_inverse)
e_arr_filtered = filter(e_arr, adder_arr_inverse)
xe_arr_filtered = filter(xe_arr, adder_arr_inverse)
ye_arr_filtered = filter(ye_arr, adder_arr_inverse)
n_cluster_arr_filtered = filter(n_cluster_arr, adder_arr_inverse)

plt.scatter(x_arr_filtered, y_arr_filtered, c=n_cluster_arr_filtered, s=e_arr_filtered, cmap=cmap, norm=norm, marker='x')
plt.scatter(xe_arr_filtered, ye_arr_filtered, c=n_cluster_arr_filtered, s=200, cmap=cmap, norm=norm, marker='1', alpha=0.2)

# removes all elements where there is only a single energy deposit, so also only shows bremsadder errors
# comment for loop if you want to see all events
for i in range(len(adder_arr)):
    if n_cluster_arr[i] >= 2:
        adder_arr[i] = False



x_arr_filtered = filter(x_arr, adder_arr)
y_arr_filtered = filter(y_arr, adder_arr)
e_arr_filtered = filter(e_arr, adder_arr)
xe_arr_filtered = filter(xe_arr, adder_arr)
ye_arr_filtered = filter(ye_arr, adder_arr)
n_cluster_arr_filtered = filter(n_cluster_arr, adder_arr)


# NOTE TO SELF: x/o = energy deposits 1/2 (upside down/right side up triangle) = electron position
plt.scatter(x_arr_filtered, y_arr_filtered, c=n_cluster_arr_filtered, s=e_arr_filtered, cmap=cmap, norm=norm)
plt.scatter(xe_arr_filtered, ye_arr_filtered, c=n_cluster_arr_filtered, s=200, cmap=cmap, norm=norm, marker='2', alpha=0.2)
plt.suptitle('x/2 = bremsadd not triggered, o/1 = bremsadd triggered')
plt.title('colour coding based on number of recorded energy clusters')
plt.colorbar()
fig.savefig('/home/felix/PycharmProjects/Thesis/Graphs/' + "bremsadder errors colour coded n_clusters x=noadd, o=added" + '.png')

# # following just plots the impacts without filtering/colour coding
# plt.scatter(x_arr, y_arr, s=e_arr)
# plt.title('Energy Deposits on 3_CAL from 1000 electrons')
# plt.ylabel("Displacement from centre (in mm)")
# plt.xlabel("Displacement from centre (in mm)")
plt.show()
