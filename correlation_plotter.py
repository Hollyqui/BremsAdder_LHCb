import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import root_pandas
from itertools import compress

# helper function
def filter(x, bool_list):
    return list(compress(x, bool_list))

# data import
filename = "/home/felix/PycharmProjects/Thesis/Data/data/DaVinci_PgunElectrons_081119_1000ev.root"
df = root_pandas.read_root(filename)
print(list(df.columns))
df_numpy = np.array(df)

# correlation that is to be checked
a = df['eminus_TRUEPT']
b = df['eminus_P']


# two array of which the correlation is to be checked
n_clusters = df['N_ECAL_clusters']
added = df['eminus_HasBremAdded']

levels = [1, 2, 3, 4, 5, 6]
colors = ['red', 'brown', 'yellow', 'green', 'blue']
cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
plt.figure()

# filter depending on whether bremsadder has been triggered
added_inverse = [not i for i in added]
a_filtered = filter(a, added_inverse)
b_filtered = filter(b,added_inverse)
n_clusters_filtered = filter(n_clusters, added_inverse)
plt.scatter(a_filtered, b_filtered, c=n_clusters_filtered, cmap=cmap, norm=norm, marker='x')

a_filtered = filter(a, added)
b_filtered = filter(b,added)
n_clusters_filtered = filter(n_clusters, added)
plt.scatter(a_filtered, b_filtered, c=n_clusters_filtered, cmap=cmap, norm=norm)

# x = y line plotted for reference
plt.plot([1,10000],[1,10000])
plt.suptitle('x = bremsadd not triggered, o = bremsadd triggered')
plt.title('colour coding based on number of recorded energy clusters')
plt.xlabel("eminus_TRUE_PT")
plt.ylabel("eminus_PT")
plt.colorbar()
plt.show()