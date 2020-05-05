try:
    from Code.xgboosting import *
    from Code.refactored_preprocessing import *
except:
    from xgboosting import *
    from refactored_preprocessing import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBClassifier
import pandas as pd
from xgboost import XGBRegressor

filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set"
START_ROW = 30000
MAX_ROWS = 10000

model = XGBClassifier({'nthread':4}) #init model
regressor = XGBRegressor({'nthread':4}) # load data
model.load_model("C:/Users/felix/Documents/University/Thesis/BremsAdder_LHCb/classifier.model")
regressor.load_model("C:/Users/felix/Documents/University/Thesis/BremsAdder_LHCb/regressor.model")
df = pd.read_pickle(filename)
df = df[START_ROW:MAX_ROWS+START_ROW]
df = df.reset_index()
# df = df.drop(['eminus_MCphotondaught/ers_ECAL_X'], axis=1)
# df = df.drop(['eminus_MCphotondaughters_ECAL_Y'], axis=1)
cand_frame = return_candidate_frame(df)



y = np.array(cand_frame['labels'])
X = np.array(cand_frame.drop(['labels'], axis=1))
# X_train, X_test, y_train, y_test = train_test_split(X,y)
# model = train_classifier(X=X_train, y=y_train, silent=False,
#                          scale_pos_weight=1, learning_rate=0.1, colsample_bytree = 0.4,
#                          subsample = 0.8, objective='binary:logistic', n_estimators=100,
#                          reg_alpha = 0.3, max_depth=8, gamma=1, early_stopping_rounds = 1000)
get_metrics(model, X, y)

calc_frame = return_calculation_frame(model, cand_frame, df)
# calc_frame


y_reg = np.array(calc_frame['labels'])
X_reg = np.array(calc_frame.drop(['labels'], axis=1))
# X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg,y_reg)
# regressor = train_regressor(X=X_train_reg, y=y_train_reg,
#                             scale_pos_weight=1, learning_rate=0.1, colsample_bytree = 0.4,
#                             subsample = 0.8, n_estimators=1000, min_child_weight=0,
#                             reg_alpha = 0.3, max_depth=6, gamma=1, early_stopping_rounds = 10000)
# X_reg
preds = regressor.predict(X_reg)
# preds
# plt.hist(y_test_reg-preds, bins=250)
# plt.hist(y_test_reg, bins=250)
# df.columns
# compare to BremsAdder_LHCb
P_REC = df['eminus_P']
P_TRUE = np.sqrt(df['eminus_TRUEP_X']**2+df['eminus_TRUEP_Y']**2+df['eminus_TRUEP_Z']**2)



shrew =preds-y_reg
brem = P_REC-P_TRUE

print("Shrew:", np.average(np.abs(shrew)))
print("Adder:", np.average(np.abs(brem)))
plt.hist(brem, bins=1000, color='red', alpha=0.5,range=(-30000,30000))
plt.hist(shrew, bins=1000, color='green', alpha=0.5,range=(-30000,30000))
plt.show()


plt.plot([1,1000000],[1,1000000])
plt.scatter(P_TRUE[:MAX_ROWS], P_TRUE[:MAX_ROWS], c='green')
plt.scatter(P_TRUE[:MAX_ROWS], P_REC[:MAX_ROWS], c='red')
# plt.scatter(P_TRUE, P_ORIG, c='black', marker='x')
plt.show()
brems_stats = pd.DataFrame({'P_TRUE': P_TRUE, 'P_REC': P_REC})
brems_stats.to_csv("C:/Users/felix/Documents/University/Thesis/brems_stats")

plt.plot([1,1000000],[1,1000000])
plt.scatter(y_reg, y_reg, c='green')
plt.scatter(y_reg, preds, c='red')
# plt.scatter(shrew_true, P_ORIG, c='black', marker='x')
plt.show()
brems_stats.to_csv("C:/Users/felix/Documents/University/Thesis/brems_stats")


plt.plot([1,1000000],[1,1000000])
plt.scatter(y_reg, preds, s=30, c='green', alpha=0.5, marker='x')
plt.scatter(P_TRUE[START_ROW:MAX_ROWS+START_ROW], P_REC[START_ROW:MAX_ROWS+START_ROW], s=30, c='red', alpha=0.5, marker='3')
plt.show()


# cand_frame

n_candidates = 10
showermax = 12650
magnet_start = 2500
magnet_end = 8000
# def plot_cand_frame():
for i in range(100):
    x_pos = []
    y_pos = []
    ph_origin_x = []
    ph_origin_y = []
    ph_origin_z = []
    daughter_colour = []
    alpha = []
    for j in range(13*n_candidates):
        if j % 13==1:
            if calc_frame.iloc[i][j]>=0.5:
                daughter_colour.append("green")
                alpha.append(1)
            else:
                daughter_colour.append("blue")
                alpha.append(0)

        if j % 13==2:
            x_pos.append(calc_frame.iloc[i][j])
        if j % 13==3:
            y_pos.append(calc_frame.iloc[i][j])
        if j % 13==10:
            ph_origin_x.append(calc_frame.iloc[i][j])
        if j % 13==11:
            ph_origin_y.append(calc_frame.iloc[i][j])
        if j % 13==12:
            ph_origin_z.append(calc_frame.iloc[i][j])

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(-4000,4000,100)
    y = np.arange(-4000,4000,100)
    Z = np.ones((80,80))*showermax
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X, Y,Z, alpha=0.1, color='blue')
    Z = np.ones((80,80))*magnet_start
    ax.plot_surface(X, Y,Z, alpha=0.1, color='yellow')
    Z = np.ones((80,80))*magnet_end
    ax.plot_surface(X, Y,Z, alpha=0.1, color='yellow')

    for j in range(len(df['eminus_MCphotondaughters_ECAL_X'][i])):
        ax.plot([df['eminus_MCphotondaughters_orivx_X'][i][j],df['eminus_MCphotondaughters_ECAL_X'][i][j]],[df['eminus_MCphotondaughters_orivx_Y'][i][j],df['eminus_MCphotondaughters_ECAL_Y'][i][j]],[df['eminus_MCphotondaughters_orivx_Z'][i][j],showermax], c='orange')
        ax.scatter(df['eminus_MCphotondaughters_orivx_X'][i][j],df['eminus_MCphotondaughters_orivx_Y'][i][j],df['eminus_MCphotondaughters_orivx_Z'][i][j], c='orange', marker='x', s=50)
        ax.scatter(df['eminus_MCphotondaughters_ECAL_X'][i][j],df['eminus_MCphotondaughters_ECAL_Y'][i][j],showermax, c='orange', marker='x')
        ax.scatter(df['eminus_ECAL_x'][i], df['eminus_ECAL_y'][i],showermax, c='blue')
    for j in range(len(x_pos)):
        ax.scatter(x_pos[j], y_pos[j], zs=showermax, zdir='z', c=daughter_colour[j], alpha=alpha[j])
    # for j in range(len(x_pos)):
        ax.scatter(ph_origin_x[j], ph_origin_y[j], ph_origin_z[j], c=daughter_colour[j], alpha=alpha[j])
    # for j in range(len(x_pos)):
        ax.plot([x_pos[j],ph_origin_x[j]], [y_pos[j],ph_origin_y[j]], [showermax,ph_origin_z[j]], c=daughter_colour[j], alpha=alpha[j])



    plt.show()
