import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import root_pandas
from itertools import compress
import seaborn as sns

# helper functions
def normalize(x):
    return np.divide(x, np.max(x))

def filter(x, bool_list):
    return list(compress(x, bool_list))

# file import
# filename = "/home/felix/PycharmProjects/Thesis/Data/data/DaVinci_PgunElectrons_081119_1000ev.root"
filename = "/home/felix/PycharmProjects/Thesis/Data/data/full_events_2k.root"
df = root_pandas.read_root(filename)
print(list(df.columns))
df_numpy = np.array(df)


fig = plt.figure()
levels = [1, 2, 3, 4, 5, 6]
colors = ['red', 'brown', 'orange', 'green', 'blue']
cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)

x_arr = []
y_arr = []
n_cluster_arr = []

for i in range(0, min(len(df), 10000)):
    # print(df['ECAL_cluster_x_arr'][i])
    for j in range(len(df['ECAL_cluster_x_arr'][i])):
        n_cluster_arr.append(df['eminus_MCphotondaughters_N'][i])
        x_arr.append(df['eminus_ECAL_x'][i]-df['ECAL_cluster_x_arr'][i][j])
        # print(df['eminus_ECAL_x'][i], df['ECAL_cluster_x_arr'][i][j])
        y_arr.append(df['eminus_ECAL_y'][i]-df['ECAL_cluster_y_arr'][i][j])
    print(x_arr[-1])
plt.scatter(x_arr, y_arr, c='black', s=10, marker='o')


x_arr = []
y_arr = []
n_cluster_arr = []


for i in range(0, min(len(df), 10000)):
    # print(df['ECAL_cluster_x_arr'][i])
    for j in range(len(df['eminus_MCphotondaughters_ECAL_X'][i])):
        n_cluster_arr.append(df['eminus_MCphotondaughters_N'][i])
        x_arr.append(df['eminus_ECAL_x'][i]-df['eminus_MCphotondaughters_ECAL_X'][i][j])
        # print(df['eminus_ECAL_x'][i], df['ECAL_cluster_x_arr'][i][j])
        y_arr.append(df['eminus_ECAL_y'][i]-df['eminus_MCphotondaughters_ECAL_Y'][i][j])
    print(x_arr[-1])
    # plt.scatter(xe_arr, ye_arr, c=n_cluster_arr, s=30, marker='1')
plt.scatter(x_arr, y_arr, c=n_cluster_arr, cmap=cmap, norm=norm, s=10, marker='o')



axes = plt.gca()
axes.set_xlim([-4000, 4000])
axes.set_ylim([-4000, 4000])
plt.title('Deposits from one electron')
plt.colorbar()
plt.show()


# plt.figure()
# compression = 50
# x_range = np.max(x_arr)-np.min(x_arr)
# y_range = np.max(y_arr)-np.min(y_arr)
# print(x_range)
# print(y_range)
# heatmap = np.zeros((int(y_range//compression),int(x_range//compression)))
# for i in range(len(x_arr)):
#     heat_x = int((x_arr[i]+np.max(x_arr)/2)//compression)
#     heat_y = int((y_arr[i]+np.max(y_arr)/2)//compression)
#     heatmap[heat_y][heat_x]+=1
# # heatmap = np.log(heatmap)
# ax = sns.heatmap(heatmap, linewidth=0.5)
# plt.show()
