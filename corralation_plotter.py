import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import root_pandas
from itertools import compress


def filter(x, bool_list):
    return list(compress(x, bool_list))

filename = "/home/felix/PycharmProjects/Thesis/Data/data/DaVinci_PgunElectrons_081119_1000ev.root"
df = root_pandas.read_root('/home/felix/PycharmProjects/Thesis/Data/data/DaVinci_PgunElectrons_081119_1000ev.root')
print(list(df.columns))
df_numpy = np.array(df)

a = df['eminus_TRUEPT']
b = df['eminus_PT']
n_clusters = df['N_ECAL_clusters']
added = df['eminus_HasBremAdded']

levels = [1, 2, 3, 4, 5, 6]
colors = ['red', 'brown', 'yellow', 'green', 'blue']
cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
plt.figure()
# filter depending on whether bremsadded has added stuff
# added = [not i for i in added]
# a_filtered = filter(a, added)
# b_filtered = filter(b,added)
# n_clusters_filtered = filter(n_clusters, added)
# plt.scatter(a_filtered, b_filtered, c=n_clusters_filtered, cmap=cmap, norm=norm, marker='x')

# added = [not i for i in added]
a_filtered = filter(a, added)
b_filtered = filter(b,added)
n_clusters_filtered = filter(n_clusters, added)
plt.scatter(a_filtered, b_filtered, c=n_clusters_filtered, cmap=cmap, norm=norm)

plt.plot([1,10000],[1,10000])
plt.legend(['x = bremsadd not active, o = bremsadd active'])
plt.xlabel("eminus_TRUE_PT")
plt.ylabel("eminus_PT")
plt.colorbar()
plt.show()