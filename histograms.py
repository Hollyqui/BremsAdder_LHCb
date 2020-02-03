import ROOT # To read .root files and transform them into pandas dataframes.
import uproot
import root_pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # To list and select the data we want for the neural network.

filename = "/home/felix/PycharmProjects/Thesis/Data/data/DaVinci_PgunElectrons_081119_1000ev.root"

df = root_pandas.read_root('/home/felix/PycharmProjects/Thesis/Data/data/DaVinci_PgunElectrons_081119_1000ev.root')
#
# kaon_tree = file["DzTree_Kaon/DecayTree"] # Select data to pick variables from.
# kaon_dataframe = kaon_tree.pandas.df() # Turns all data into a dataframe that pandas can work with.

df_numpy = np.array(df, ndmin=2)
for i in range(1,37):
    column = df.columns.values[i]
    if(i == 12 or i == 13 or i == 14):
        print(i, "-", column)
        arr = []
        for j in range(len(df_numpy)):            # print(df_numpy[j,i])
            for k in df_numpy[j,i]:
                arr.append(k)
        fig = plt.figure()
        plt.hist(arr, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram of value: " + column)
        fig.savefig('/home/felix/PycharmProjects/Thesis/Graphs/graph' + column + '.png')
    elif(i != 34):
        print(i, "-", column)
        a = df_numpy[:,i]
        fig = plt.figure()
        plt.hist(a, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram of value: " + column)
        # plt.show()
        fig.savefig('/home/felix/PycharmProjects/Thesis/Graphs/graph'+column+'.png')