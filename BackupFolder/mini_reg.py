import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
# from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
# !conda install tensorflow
# import tensorflow
def plot_hist(x, train=None):
    if train=='train':
        plt.hist(calc_frame['lin_pred']-x-labels_train, bins=1000, color='blue', alpha=0.5,range=(-30000,30000))
    elif train=='test':
        plt.hist(calc_frame_test['lin_pred']-x-labels_test, bins=1000, color='yellow', alpha=0.5, range=(-30000,30000))
    else:
        print("specify train/test")

#%%


################### DATASET LOADING ##################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns

################### DATASET LOADING ##################################
# try:
calc_frame = pd.read_pickle("calc_frame_10k_20k")
# except:
    # calc_frame = pd.read_pickle("C:/Users/szymo/Documents/Felix's help/calc_frame_10k_20k")
calc_frame.fillna(0, inplace=True)
# drop labels with value 0
zero_labels = calc_frame[ calc_frame['labels'] == 0 ].index
calc_frame.drop(zero_labels , inplace=True)



#%%
#add column with linear regression predicitons
p_phot = np.array(abs(calc_frame.iloc[:,6])+abs(calc_frame.iloc[:,6+7])+abs(calc_frame.iloc[:,6+14]), ndmin=2).T
p_phot_test = np.array(abs(calc_frame_test.iloc[:,6])+abs(calc_frame_test.iloc[:,6+7])+abs(calc_frame_test.iloc[:,6+14]), ndmin=2).T



# lin_reg = LinearRegression().fit(p_phot, calc_frame['labels']-calc_frame['eminus_P'])
# lin_reg.score(p_phot_test, calc_frame_test['labels']-calc_frame_test['eminus_P'])
# lin_reg.score(p_phot, calc_frame['labels']-calc_frame['eminus_P'])
# lin_shrew_test = lin_reg.predict(p_phot_test)
# calc_frame['lin_pred'] = lin_reg.predict(p_phot)
calc_frame['lin_pred'] = p_phot

plt.scatter(np.array(calc_frame['labels']-calc_frame['eminus_nobrem_P']),np.array(p_phot), s=5)
plt.xlabel('eminus_TRUEP-eminus_nobrem_P')
plt.ylabel('Sum of PT * DaughterPhoton Likelihood')

#%% continue


# calc_frame.columns.tolist()
y_train = np.array(calc_frame['lin_pred']+calc_frame['eminus_P'])-np.array(calc_frame['labels'])
y_train = y_train.astype(np.float)


labels_train = np.array(calc_frame['labels'])
# X_train = np.array(calc_frame)*1
# X_train = np.array(calc_frame.drop(['labels','eminus_P'], axis=1))*1
# calc_frame.columns
# calc_frame['velo_ttrack_angle']
# calc_frame['lin_pred']
X_train = np.array(calc_frame.drop(['labels'],axis=1))*1

X_train = X_train.astype(np.float)

# try:
calc_frame_test = pd.read_pickle("calc_frame_20k_30k")
# except:
    # calc_frame_test = pd.read_pickle("C:/Users/szymo/Documents/Felix's help/calc_frame_20k_30k")
calc_frame_test.fillna(0, inplace=True)

zero_labels = calc_frame_test[ calc_frame_test['labels'] == 0 ].index
calc_frame_test.drop(zero_labels , inplace=True)


# calc_frame_test['lin_pred'] = lin_shrew_test
calc_frame_test['lin_pred'] = p_phot_test

pd.set_option('display.max_columns', 1000)
# calc_frame
y_test = np.array(calc_frame_test['lin_pred']+calc_frame_test['eminus_P'])-np.array(calc_frame_test['labels'])
y_test = y_test.astype(np.float)

labels_test= np.array(calc_frame_test['labels'])
# X_test = np.array(calc_frame_test)*1
# X_test = np.array(calc_frame_test.drop(['labels','eminus_P'], axis=1))*1
X_test = np.array(calc_frame_test.drop(['labels'],axis=1))*1
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



model= xgb.XGBRegressor(objective='reg:squarederror', verbosity=2)
# model = xgb.XGBRegressor(objective='reg:linear', max_depth=8, learning_rate=0.3, batch_size=32, verbosity=2, n_estimators=1000, min_child_weight=10,
#                          reg_alpha=0.2, reg_lambda=0.3, subsample=0.8, gamma=100000000)
# reg_cv = GridSearchCV(gbm, {"colsample_bytree":[1.0],"min_child_weight":[0.4,1.2]
#                             ,'max_depth': [4,6,8], 'n_estimators': [500,1000]}, verbose=2)
# reg_cv.fit(X_train,y_train)
# reg_cv.best_params_

# model = xgb.XGBRegressor(**reg_cv.best_params_)
model.fit(X_train,y_train)

# calc_frame_test

xgb_shrew_test = model.predict(X_test)
xgb_shrew_train = model.predict(X_train)
xgb.plot_importance(model)
# plt.hist(calc_frame['eminus_nobrem_P']+calc_frame['lin_pred']-calc_frame['labels']-2300, bins=1000, range=(-3e4,3e4), alpha=0.5)
a = np.sum(np.abs(calc_frame['eminus_nobrem_P']+calc_frame['lin_pred']-calc_frame['labels']-2300))/10000
b = np.sum(np.abs(calc_frame['eminus_nobrem_P']+calc_frame['lin_pred']-xgb_shrew_train-calc_frame['labels']))/10000
c = np.sum(np.abs(calc_frame_test['eminus_nobrem_P']+calc_frame_test['lin_pred']-xgb_shrew_test-calc_frame_test['labels']))/10000
d = np.sum(np.abs(calc_frame_test['eminus_nobrem_P']+calc_frame_test['lin_pred']-calc_frame_test['labels']-2300))/10000
e = np.sum(np.abs(calc_frame_test['eminus_P']-calc_frame_test['labels']))/10000
a-b
a
b
c
d
e

plt.hist(calc_frame['eminus_nobrem_P']+calc_frame['lin_pred']-xgb_shrew_train-calc_frame['labels'], bins=1000, range=(-3e4,3e4))

plt.hist(calc_frame_test['eminus_nobrem_P']+calc_frame_test['lin_pred']-xgb_shrew_test-calc_frame_test['labels'], bins=1000, range=(-3e4,3e4))

plot_hist(xgb_shrew_test, 'test')

xgb_shrew_train[0]
xgb_shrew_test[0]

############################# GPR ###########################################
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, RationalQuadratic, ConstantKernel
# kernel = DotProduct()#1**2*RBF(length_scale_bounds = (1e-1, 1e9))
kernel = RBF(1e6, (1e-15, 1e15))*ConstantKernel(1e7, (1e-15, 1e15))
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=10, optimizer='fmin_l_bfgs_b',n_restarts_optimizer=3)
gpr =gpr.fit(X=X_train[:1000], y=y_train[:1000])
gpr.score(X_train, y_train)
gpr.score(X_test,y_test)
gpr_shrew_train = gpr.predict(X_train)
gpr_shrew_test = gpr.predict(X_test)
gpr.kernel_.get_params()
plt.hist(calc_frame['eminus_nobrem_P']+calc_frame['lin_pred']-gpr_shrew_train-calc_frame['labels'], bins=1000, range=(-3e4,3e4))

plt.hist(calc_frame_test['eminus_nobrem_P']+calc_frame_test['lin_pred']-gpr_shrew_test-calc_frame_test['labels'], bins=1000, range=(-3e4,3e4))


plt.hist(calc_frame_test['eminus_P']-calc_frame_test['labels'], bins=1000, range=(-3e4,3e4))


#################################### lin reg ####################################

lin_reg = LinearRegression().fit(X_train, y_train)
lin_reg.score(X_train,y_train)
lin_reg.score(X_test, y_test)
lin_shrew_test = lin_reg.predict(X_test)

plt.hist(calc_frame_test['eminus_P']+calc_frame_test['lin_pred']-lin_shrew_test-calc_frame_test['labels'], bins=1000, range=(-3e4,3e4))



np.average(np.abs(calc_frame_test['eminus_P']-calc_frame_test['labels']))
np.average(np.abs(calc_frame_test['eminus_P']+calc_frame_test['lin_pred']-lin_shrew_test-calc_frame_test['labels']))
