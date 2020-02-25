import root_pandas
import numpy as np
import matplotlib.pyplot as plt

# Import Files
filename = "/home/felix/PycharmProjects/Thesis/Data/data/DaVinci_PgunElectrons_081119_1000ev.root"
df = root_pandas.read_root(filename)
print(list(df.columns))

# creates all histograms TODO: outlier removal
# df_numpy = np.array(df, ndmin=2)
# for i in range(1,37):
#     column = df.columns.values[i]
#     if(i == 12 or i == 13 or i == 14):
#         print(i, "-", column)
#         arr = []
#         for j in range(len(df_numpy)):            # print(df_numpy[j,i])
#             for k in df_numpy[j,i]:
#                 arr.append(k)
#         fig = plt.figure()
#         plt.hist(arr, bins='auto')  # arguments are passed to np.histogram
#         plt.title("Histogram of value: " + column)
#         fig.savefig('/home/felix/PycharmProjects/Thesis/Graphs/graph' + column + '.png')
#     elif(i != 34 and i!=23):
#         print(i, "-", column)
#         a = df_numpy[:,i]
#         fig = plt.figure()
#         plt.hist(a, bins='auto')  # arguments are passed to np.histogram
#         plt.title("Histogram of value: " + column)
#         # plt.show()
#         fig.savefig('/home/felix/PycharmProjects/Thesis/Graphs/graph'+column+'.png')


column = 'eminus_P'
P_REC = df[column]
column = 'eminus_TRUEP_X'
x = np.array(df[column])
column = 'eminus_TRUEP_Y'
y = np.array(df[column])
column = 'eminus_TRUEP_Z'
z = np.array(df[column])
TRUE_P = np.sqrt(x**2+y**2+z**2)
plottable = (TRUE_P-P_REC)
# plottable[plottable]
plt.hist(plottable, bins=5000)  # arguments are passed to np.histogram
plt.title("Histogram of value: " + "P_REC - TRUE_P")
plt.show()