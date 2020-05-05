import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

################### DATASET LOADING ##################################
# try:
calc_frame = pd.read_pickle("calc_frame_10k_20k")
# except:
#     calc_frame = pd.read_pickle("C:/Users/szymo/Documents/Felix's help/calc_frame_10k_20k")
calc_frame.fillna(0, inplace=True)

# drop labels with value 0
zero_labels = calc_frame[ calc_frame['labels'] == 0 ].index
calc_frame.drop(zero_labels , inplace=True)

# calc_frame.columns.tolist()
y_train = np.array(calc_frame['eminus_nobrem_P'])-np.array(calc_frame['labels'])
y_train = y_train.astype(np.float)


labels_train = np.array(calc_frame['labels'])
X_train = np.array(calc_frame)*1
# X_train = np.array(calc_frame.drop(['labels'], axis=1))*1

X_train = X_train.astype(np.float)
# should be all 0
# calc_frame['eminus_nobrem_P']-y_train-labels_train


# pca = PCA(n_components=2)
# pca.fit(X_train)
# X_pca = pca.transform(X_train)
# X_pca.shape
#
# fig = plt.figure(figsize=(12,12))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2])
# plt.show()

# try:
calc_frame_test = pd.read_pickle("calc_frame_20k_30k")
# except:
#     calc_frame_test = pd.read_pickle("C:/Users/szymo/Documents/Felix's help/calc_frame_20k_30k")
calc_frame_test.fillna(0, inplace=True)

zero_labels = calc_frame_test[ calc_frame_test['labels'] == 0 ].index
calc_frame_test.drop(zero_labels , inplace=True)

pd.set_option('display.max_columns', 1000)
# calc_frame
y_test = np.array(calc_frame_test['eminus_nobrem_P'])-np.array(calc_frame_test['labels'])
y_test = y_test.astype(np.float)

labels_test= np.array(calc_frame_test['labels'])
X_test = np.array(calc_frame_test)*1
# X_test = np.array(calc_frame_test.drop(['labels'], axis=1))*1
X_test = X_test.astype(np.float)
# should also be all 0
# calc_frame_test['eminus_nobrem_P']-y_test-labels_test

# normalizer = Normalizer().fit(X=X_train)
#
# X_train = normalizer.transform(X_train)
# X_test = normalizer.transform(X_test)

scaler = StandardScaler()

# scaler.fit(np.concatenate((X_train, X_test)))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# for i in range(len(X_train[0])):
#     print(calc_frame.columns[i])
#     plt.figure()
#     plt.hist(X_train[:,i], bins=100, color='red', alpha=0.5)
#     plt.hist(X_test[:,i], bins=100, color='green', alpha=0.5)
#     # plt.savefig("historgram"+str(calc_frame.columns[i]))
#     plt.show()


###################################################################################################################
# for i in range(min(len(X_train[0]),1000)):
#     plot = sns.jointplot(x=X_train[:,i], y=y_train);



plt.hist(calc_frame['eminus_P']-calc_frame['labels'], bins=1000, range=(-20000,20000), alpha=0.5, color='blue')
plt.hist(calc_frame['eminus_nobrem_P']-calc_frame['labels'], bins=1000, range=(-30000,30000), alpha=0.4, color='black')
# plt.hist(calc_frame['eminus_nobrem_P']+calc_frame.iloc[:,6]+calc_frame.iloc[:,6+7]+calc_frame.iloc[:,6+14]-calc_frame['labels'], bins=1000, range=(-20000,20000), alpha=0.5, color='green')
# plt.show()

plt.hist(calc_frame['eminus_nobrem_P']+ abs(calc_frame.iloc[:,6])+abs(calc_frame.iloc[:,6+7])+abs(calc_frame.iloc[:,6+14])*100-calc_frame['labels'], bins=1000, range=(-20000,20000), alpha=0.5, color='orange')
plt.show()


np.min(abs(calc_frame.iloc[:,6])+abs(calc_frame.iloc[:,6+7])+abs(calc_frame.iloc[:,6+14]))
plt.scatter(calc_frame['labels']-calc_frame['eminus_nobrem_P'], abs(calc_frame.iloc[:,6])+abs(calc_frame.iloc[:,6+7])+abs(calc_frame.iloc[:,6+14]), marker='1', s=1)
plt.show()
plt.scatter(calc_frame['labels']-calc_frame['eminus_nobrem_P'], abs(calc_frame.iloc[:,6])*abs(calc_frame.iloc[:,5])+abs(calc_frame.iloc[:,6+7])*abs(calc_frame.iloc[:,5+7])+abs(calc_frame.iloc[:,6+14])*abs(calc_frame.iloc[:,5+14]), marker='1', s=1)
plt.show()

abs(calc_frame.iloc[:,6])
abs(calc_frame.iloc[:,5])

plt.scatter(calc_frame['labels']-calc_frame['eminus_nobrem_P'], abs(calc_frame.iloc[:,0]*calc_frame.iloc[:,3])+abs(calc_frame.iloc[:,0+7]*calc_frame.iloc[:,3+7])+abs(calc_frame.iloc[:,0+14]*calc_frame.iloc[:,3+14]), marker='1', s=1)
plt.show()

calc_frame.columns
plt.scatter(calc_frame['labels']-calc_frame['eminus_nobrem_P'], calc_frame['velo_ttrack_angle'], marker='1', s=1)
plt.show()


from Code.xgboosting import *


cand_frame = pd.read_pickle("cand_frame_10k_20k")
cand_frame.columns.tolist()
model = XGBClassifier({'nthread':4}) #init model

model.load_model("C:/Users/felix/Documents/University/Thesis/BremsAdder_LHCb/classifier.model")
xgb.plot_importance(model)
