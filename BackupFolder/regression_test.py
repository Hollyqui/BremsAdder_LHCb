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
        plt.hist(x-labels_train, bins=1000, color='blue', alpha=0.5,range=(-30000,30000))
    elif train=='test':
        plt.hist(x-labels_test, bins=1000, color='orange', alpha=0.5, range=(-30000,30000))
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
calc_frame = pd.read_pickle("calc_frame_down")
calc_frame_test = pd.read_pickle("calc_frame_20k_30k")

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

lin_reg = LinearRegression().fit(p_phot, calc_frame['labels']-calc_frame['eminus_nobrem_P'])
lin_reg.score(p_phot_test, calc_frame_test['labels']-calc_frame_test['eminus_nobrem_P'])
lin_reg.score(p_phot, calc_frame['labels']-calc_frame['eminus_nobrem_P'])
# lin_shrew_test = lin_reg.predict(p_phot_test)
calc_frame['lin_pred'] = lin_reg.predict(p_phot)
calc_frame_test['lin_pred'] = lin_reg.predict(p_phot_test)


#%% continue


# calc_frame.columns.tolist()
y_train = np.array(calc_frame['labels'])
y_train = y_train.astype(np.float)


labels_train = np.array(calc_frame['labels'])
# X_train = np.array(calc_frame)*1
X_train = np.array(calc_frame.drop(['labels'], axis=1))*1

X_train = X_train.astype(np.float)

# try:
# except:
    # calc_frame_test = pd.read_pickle("C:/Users/szymo/Documents/Felix's help/calc_frame_20k_30k")
calc_frame_test.fillna(0, inplace=True)

zero_labels = calc_frame_test[ calc_frame_test['labels'] == 0 ].index
calc_frame_test.drop(zero_labels , inplace=True)


# calc_frame_test['lin_pred'] = lin_shrew_test


pd.set_option('display.max_columns', 1000)
# calc_frame
y_test = np.array(calc_frame_test['labels'])
y_test = y_test.astype(np.float)

labels_test= np.array(calc_frame_test['labels'])
# X_test = np.array(calc_frame_test)*1
X_test = np.array(calc_frame_test.drop(['labels'], axis=1))*1
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



#%%
############################# GAUSSIAN PROCESS ###############################

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, RationalQuadratic, ConstantKernel, ExpSineSquared
kernel = ConstantKernel()*1**2*RationalQuadratic(length_scale=1e3)#1**2*Matern(length_scale=1)#+RBF(length_scale=1)
# kernel = 1**2*RBF(1e-4, (1e-7, 1e5))+1**2*ConstantKernel(1, (1e-5, 1e5))
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=5e2, optimizer='fmin_l_bfgs_b',n_restarts_optimizer=10)
gpr =gpr.fit(X=calc_frame.drop(['labels'], axis=1)[1000:2000], y=y_train[1000:2000])
gpr.score(X_train, y_train)
gpr.score(X_test,y_test)
gpr_shrew_train = gpr.predict(X_train)
gpr_shrew_test = gpr.predict(X_test)
gpr.kernel_.get_params()
plot_hist(gpr_shrew_train, 'train')
plot_hist(0,'test')
plot_hist(gpr_shrew_test, 'test')
plot_hist(calc_frame['eminus_P'], 'train')

# plt.xlim(-200000,0)
# plt.ylim(-200000,0)

plt.scatter(y_test, gpr_shrew_test, s=1)
# plt.xlim(-200000,0)
# plt.ylim(-200000,0)

# y_train
train_preds
preds

#%%
############################# XGBOOST ####################################

# model= xgb.XGBRegressor(verbosity=2)
model = xgb.XGBRegressor(objective='reg:linear', max_depth=8, learning_rate=0.3, batch_size=32, verbosity=2, n_estimators=1000, min_child_weight=10,
                         reg_alpha=0.2, reg_lambda=0.3, subsample=0.8, gamma=100000000)
# reg_cv = GridSearchCV(gbm, {"colsample_bytree":[1.0],"min_child_weight":[0.4,1.2]
#                             ,'max_depth': [4,6,8], 'n_estimators': [500,1000]}, verbose=2)
# reg_cv.fit(X_train,y_train)
# reg_cv.best_params_

# model = xgb.XGBRegressor(**reg_cv.best_params_)
model.fit(calc_frame.drop(['labels'], axis=1),y_train)

# calc_frame_test

xgb_shrew_test = model.predict(calc_frame_test.drop(['labels'], axis=1))
xgb_shrew_train = model.predict(calc_frame.drop(['labels'], axis=1))
xgb.plot_importance(model)
plot_hist(xgb_shrew_train, 'train')
plot_hist(xgb_shrew_test, 'test')

plot_hist(calc_frame['eminus_P'], 'train')


# gpr_shrew_train
# gpr_shrew_test
# y_test

plt.scatter(y_train, xgb_shrew_train, s=1)
plt.scatter(y_test, xgb_shrew_test, s=1)


#%%
########################## SKLEARN NEURAL NET #############################

X_all = np.concatenate((X_test,X_train))
y_all = np.concatenate((y_test,y_train))
calc_frame.columns.tolist()
X2_train, X2_test, y2_train, y2_test = train_test_split(X_all, y_all, test_size=0.33, random_state=42)

nn = MLPRegressor(hidden_layer_sizes=(64), max_iter=1000, verbose=True, validation_fraction=0.2, n_iter_no_change=10,
                  activation='relu', learning_rate_init=0.05, batch_size=128, solver='adam', learning_rate='constant', alpha=0.01)
nn.fit(X=calc_frame.drop(['labels'], axis=1),y=y_train)
nn.get_params()
nn.score(X_train,y_train)
nn.score(X_test,y_test)
nn_preds_train = np.array(nn.predict(X_train))
nn_preds_train.shape
nn_preds_test = nn.predict(X_test)


plt.scatter(y_train, nn_preds_train, s=1)
plt.scatter(y_test, nn_preds_test, s=1)

#%%

###################### PYTORCH NEURAL NETWORK ###########################


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

inputDim = len(X_train[0])       # takes variable 'x'
outputDim = 1       # takes variable 'y'
learningRate = 0.05
epochs = 100

model = linearRegression(inputDim, outputDim).float()
##### For GPU #######
if torch.cuda.is_available():
    model.cuda()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)


for epoch in range(epochs):
    # Converting inputs and labels to Variable
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(X_train).float().cuda())
        labels = Variable(torch.from_numpy(y_train).float().cuda())
    else:
        inputs = Variable(torch.from_numpy(X_train).float())
        labels = Variable(torch.from_numpy(y_train).float())

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))


torch_train = model(Variable(torch.from_numpy(X_train).float())).detach().numpy()[0]
torch_train.tolist()
torch_test = model(Variable(torch.from_numpy(X_test).float())).detach().numpy()[0]
X_train

#%%

################ SVM ##################################

svr_rbf = SVR(kernel='rbf', C=2000000000, gamma='auto', epsilon=500, verbose=True)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=3000000000, gamma='auto', degree=10, epsilon=.1,
               coef0=1)

svr_poly.fit(X_train, y_train)
svm_shrew_train = svr_poly.predict(X_train)
svm_shrew_test = svr_poly.predict(X_test)

svm_shrew_train
# svm_shrew_train0-p.p;;;;;;;;/

%%

#%%
type(calc_frame.iloc[:,126][0])
type(calc_frame.iloc[:,127][0])


############### LINEAR REGRESSOR #################################
# calc_frame.columns.tolist()
# frame127plus = calc_frame.iloc[:,220:]
# # frame125 = calc_frame.iloc[:,97]
# frame = frame127plus.join(frame125)


lin_reg = LinearRegression().fit(X_train, labels_train)
lin_reg.score(X_train, labels_train)
lin_reg.score(X_test, labels_test)
lin_shrew_train = lin_reg.predict(X_train)
lin_shrew_test = lin_reg.predict(X_test)

lin_reg.coef_


#%%


##### TESTING ############


def combine(gpr_arr, xgb_arr, dist):
    combined_arr = []
    for i in range(len(gpr_arr)):
        if np.abs(gpr_arr[i]-xgb_arr[i])>dist:
            combined_arr.append(xgb_arr[i])
        else:
            combined_arr.append(gpr_arr[i])
    return combined_arr
# filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set"
# START_ROW = 20000
# MAX_ROWS = 10000
# df = pd.read_pickle(filename)
# df = df[START_ROW:MAX_ROWS+START_ROW]
# df = df.reset_index()
#
#
# P_REC = df['eminus_P']
# P_TRUE = np.sqrt(df['eminus_TRUEP_X']**2+df['eminus_TRUEP_Y']**2+df['eminus_TRUEP_Z']**2)
#
#
# # gpr_shrew = gpr_preds
# brem = P_REC-P_TRUE
# combined_shrew = (gpr_shrew+xgb_shrew)/2
# np.max(nn_preds_train)
# labels_train\
#### TRAIN



average_shrew_test = (gpr_shrew_test+xgb_shrew_test+nn_preds_test)/3
average_shrew_train = (gpr_shrew_train+xgb_shrew_train+nn_preds_train)/3

print("NO Shrew", np.average(np.abs(calc_frame['eminus_nobrem_P']-labels_train)))
print("LIN Shrew", np.average(np.abs(lin_shrew_train-labels_train)))
print("SVM Shrew", np.average(np.abs(svm_shrew_train-labels_train)))
print("GPR Shrew:", np.average(np.abs(gpr_shrew_train-labels_train)))
print("XGB Shrew:", np.average(np.abs(xgb_shrew_train-labels_train)))
print("NN Shrew:", np.average(np.abs(nn_preds_train-labels_train)))
# print("TORCH Shrew:", np.average(np.abs(torch_train-labels_train)))

print("Combined Shrew:", np.average(np.abs(average_shrew_train-y_train)))
# print("Adder:", np.average(np.abs(brem)))
print("Adder:", np.average(np.abs(calc_frame['eminus_P']-labels_train)))

#### TEST


combined_shrew_test = combine(gpr_shrew_test, xgb_shrew_test, 3000)
# print("Shrew Training:", np.average(np.abs(train_preds-y_train)))
print("LIN Shrew", np.average(np.abs((calc_frame_test['eminus_nobrem_P'])-lin_shrew_test-labels_test)))
print("SVM Shrew", np.average(np.abs((calc_frame_test['eminus_nobrem_P'])-svm_shrew_test-labels_test)))
print("GPR Shrew:", np.average(np.abs((calc_frame_test['eminus_nobrem_P'])-gpr_shrew_test-labels_test)))
print("XGB Shrew:", np.average(np.abs((calc_frame_test['eminus_nobrem_P'])-xgb_shrew_test-labels_test)))
print("NN Shrew:", np.average(np.abs((calc_frame_test['eminus_nobrem_P'])-nn_preds_test-labels_test)))
print("Average Shrew:", np.average(np.abs((calc_frame_test['eminus_nobrem_P'])-average_shrew_test-labels_test)))
print("Combined Shrew:", np.average(np.abs((calc_frame_test['eminus_nobrem_P'])-combined_shrew_test-labels_test)))

print("Adder:", np.average(np.abs(calc_frame_test['eminus_P']-labels_test)))


count=0
total=0
for i in range(len(calc_frame)):
    if calc_frame[i][0]>0.5:
        count+=1
    total+=1

np.average(np.abs(labels_test-calc_frame_test['eminus_P']))

for i in range(100):
    try:
        plt.scatter(labels_test[i], labels_test[i], c='green')
        # plt.scatter(labels_test[i], calc_frame_test['eminus_nobrem_P'][i], c='red')
        plt.scatter(labels_test[:100], calc_frame_test['eminus_P'][:100], c='red')
        #
        # plt.scatter(labels_test[i],lin_shrew_test[i], c='grey')
        # # plt.scatter(labels_test[i], calc_frame_test['eminus_nobrem_P'][i]-svm_shrew_test[i], c='blue')
        # plt.scatter(labels_test[i],gpr_shrew_test[i], c='yellow')
        # plt.scatter(labels_test[i],xgb_shrew_test[i], c='orange')
        # plt.scatter(labels_test[i],nn_preds_test[i], c='pink')
        plt.scatter(labels_test[i],average_shrew_test[i], c='black')
        # plt.scatter(labels_test[i], calc_frame_test['eminus_P'][i], c='red')
    except:
        pass






print("Adder:", np.average(np.abs(calc_frame_test['eminus_P']-labels_test)))
# plt.hist(calc_frame['eminus_nobrem_P']-labels_train, bins=1000, color='brown', alpha=0.5,range=(-30000,30000))
plt.hist(calc_frame['eminus_P']-labels_train, bins=1000, color='red', alpha=0.5,range=(-30000,30000))
plt.hist(calc_frame['eminus_nobrem_P']-xgb_shrew_train-labels_train, bins=1000, color='blue', alpha=0.5,range=(-30000,30000))
# plt.hist(calc_frame['eminus_nobrem_P']-lin_shrew_train-labels_train, bins=1000, color='black', alpha=0.5, range=(-30000,30000))
# plt.hist(calc_frame['eminus_nobrem_P']-gpr_shrew_train-labels_train, bins=1000, color='green', alpha=0.5,range=(-30000,30000))
# plt.hist(combined_shrew-y_test, bins=1000, color='orange', alpha=0.5,range=(-30000,30000))
# plt.hist(calc_frame['eminus_nobrem_P']-nn_preds_train-labels_train, bins=1000, alpha=0.5, color='orange', range=(-30000,30000))
# plt.hist(calc_frame['eminus_nobrem_P']-svm_shrew_train-labels_train, bins=1000, color='blue', alpha=0.5,range=(-30000,30000))

# (xgb_shrew_test+(svm_shrew_test-7000)+
average_shrew_test = (gpr_shrew_test+svm_shrew_test+xgb_shrew_test)/3
print("Average Shrew:", np.average(np.abs((calc_frame_test['eminus_nobrem_P'])-average_shrew_test-labels_test)))

plt.hist(calc_frame_test['eminus_P']-labels_test, bins=1000, color='red', alpha=0.5,range=(-30000,30000))
plt.hist(calc_frame_test['eminus_nobrem_P']-labels_test, bins=1000, color='brown', alpha=0.5,range=(-30000,30000))
# plt.hist(calc_frame_test['eminus_nobrem_P']-xgb_shrew_test-labels_test, bins=1000, color='black', alpha=0.5,range=(-30000,30000))
# plt.hist(calc_frame_test['eminus_nobrem_P']-svm_shrew_test-labels_test, bins=1000, color='blue', alpha=0.5,range=(-30000,30000)) # for some reason this one is moved by 7k
#
# plt.hist(calc_frame_test['eminus_nobrem_P']-lin_shrew_test-labels_test, bins=1000, color='green', alpha=0.5, range=(-30000,30000))
# plt.hist(calc_frame_test['eminus_nobrem_P']-gpr_shrew_test-labels_test-8500, bins=1000, color='green', alpha=0.5, range=(-30000,30000))
# plt.hist(calc_frame_test['eminus_nobrem_P']-average_shrew_test-labels_test, bins=1000, color='green', alpha=0.5, range=(-30000,30000))
# plt.hist(calc_frame_test['eminus_nobrem_P']-nn_preds_test-labels_test, bins=1000, alpha=0.5, color='orange', range=(-30000,30000))
# plt.hist(calc_frame_test['eminus_nobrem_P']-combined_shrew_test-labels_test, bins=1000, color='green', alpha=0.5, range=(-30000,30000))
plt.hist(calc_frame_test['lin_pred']+calc_frame_test['eminus_nobrem_P']-labels_test, bins=1000, alpha=0.5, color='orange', range=(-30000,30000))
calc_frame


#
