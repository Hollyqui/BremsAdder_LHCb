try:
    from Code.xgboosting import *
    from Code.refactored_preprocessing import *
except:
    from xgboosting import *
    from refactored_preprocessing import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

filename = "C:/Users/felix/Documents/University/Thesis/big_track_electron_set"
# df_full = pd.read_pickle( "C:/Users/felix/Documents/University/Thesis/big_track_electron_set")


#%%
######## TRAIN CLASSIFIER
try:
    cand_frame = pd.read_pickle("cand_frame_0_10k")
except:

    START_ROW = 0
    MAX_ROWS = 10000
    df = pd.read_pickle(filename)
    df = df[START_ROW:MAX_ROWS+START_ROW]
    df = df.reset_index()
    cand_frame = return_candidate_frame(df, n_candidate=5, DAUGHTER_CLUSTER_MATCH_MAX=50)
    cand_frame.to_pickle("cand_frame_0_10k")



y = np.array(cand_frame['labels'])
X = np.array(cand_frame.drop(['labels'], axis=1))
X_train, X_test, y_train, y_test = train_test_split(X,y)
model = train_classifier(X=X_train, y=y_train, silent=False,
                         scale_pos_weight=1, learning_rate=0.1, colsample_bytree = 0.4,
                         subsample = 0.8, objective='binary:logistic', n_estimators=1000,
                         reg_alpha = 0.3, max_depth=6, gamma=1, early_stopping_rounds = 1000)
get_metrics(model, X_test, y_test)

model.save_model('classifier.model')


#%%

##################### TRAIN REGRESSOR ##########################



try:
    cand_frame = pd.read_pickle("cand_frame_10k_20k")
except:

    START_ROW = 10000
    MAX_ROWS = 10000
    df = pd.read_pickle(filename)
    df = df[START_ROW:MAX_ROWS+START_ROW]
    df = df.reset_index()
    cand_frame = return_candidate_frame(df, n_candidate=5, DAUGHTER_CLUSTER_MATCH_MAX=50)
    cand_frame.to_pickle("cand_frame_10k_20k")


y = np.array(cand_frame['labels'])
X = np.array(cand_frame.drop(['labels'], axis=1))
get_metrics(model,X,y)

try:
    calc_frame = pd.read_pickle("calc_frame_10k_20k")
except:
    calc_frame = return_calculation_frame(model, cand_frame, df, n_candidate=5)
    calc_frame.to_pickle("calc_frame_10k_20k")

calc_frame.columns
# calc_frame


y_reg = np.array(calc_frame['labels'])
X_reg = np.array(calc_frame.drop(['labels'], axis=1))
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg,y_reg)

regressor = train_regressor(X=X_train_reg, y=y_train_reg,
                            scale_pos_weight=1, learning_rate=0.1, colsample_bytree = 0.4,
                            subsample = 0.9, n_estimators=1000, min_child_weight=0.4,
                            reg_alpha = 0.3, max_depth=6, gamma=1, early_stopping_rounds = 1000)

#
model = XGBRegressor(verbose=2,
                      scale_pos_weight=1, learning_rate=0.1, colsample_bytree = 0.4,
                      subsample = 0.9, n_estimators=1000, min_child_weight=0.4,
                      reg_alpha = 0.3, max_depth=6, gamma=1, early_stopping_rounds = 10)
model.fit(X_train_reg, y_train_reg)

preds = regressor.predict(X_test_reg)


shrew =preds-y_test_reg
print("Shrew:", np.average(np.abs(shrew)))

regressor.save_model('regressor.model')

############################## TEST MODELS ###########################


try:
    cand_frame = pd.read_pickle("cand_frame_20k_30k")
except:
    START_ROW = 20000
    MAX_ROWS = 10000
    df = pd.read_pickle(filename)
    df = df[START_ROW:MAX_ROWS+START_ROW]
    df = df.reset_index()

    cand_frame = return_candidate_frame(df, n_candidate=5, DAUGHTER_CLUSTER_MATCH_MAX=50)
    cand_frame.to_pickle("cand_frame_20k_30k")


try:
    pd.read_pickle("calc_frame_20k_30k")
except:
    calc_frame = return_calculation_frame(model, cand_frame, df, n_candidate=5)
    calc_frame.to_pickle("calc_frame_20k_30k")


y_reg = np.array(calc_frame['labels'])
X_reg = np.array(calc_frame.drop(['labels'], axis=1))
preds = regressor.predict(X_reg)




P_REC = df['eminus_P']
P_TRUE = np.sqrt(df['eminus_TRUEP_X']**2+df['eminus_TRUEP_Y']**2+df['eminus_TRUEP_Z']**2)

shrew =preds-y_reg
brem = P_REC-P_TRUE
print("Shrew:", np.average(np.abs(shrew)))
print("Adder:", np.average(np.abs(brem)))

plt.hist(brem, bins=1000, color='red', alpha=0.5,range=(-30000,30000))
plt.hist(shrew, bins=1000, color='green', alpha=0.5,range=(-30000,30000))
plt.show()


plt.scatter(P_TRUE, P_TRUE, c='green')
plt.scatter(P_TRUE, P_REC, c='red')
# plt.scatter(P_TRUE, P_ORIG, c='black', marker='x')
plt.show()
brems_stats = pd.DataFrame({'P_TRUE': P_TRUE, 'P_REC': P_REC})
brems_stats.to_csv("C:/Users/felix/Documents/University/Thesis/brems_stats")

plt.plot([1,1000000],[1,1000000])
plt.scatter(y_test_reg, y_test_reg, c='green')
plt.scatter(y_test_reg, preds, c='red')
# plt.scatter(shrew_true, P_ORIG, c='black', marker='x')
plt.show()
brems_stats.to_csv("C:/Users/felix/Documents/University/Thesis/brems_stats")


plt.plot([1,1000000],[1,1000000])
plt.scatter(P_TRUE[START_ROW:(MAX_ROWS//4)+START_ROW], P_REC[START_ROW:(MAX_ROWS//4)+START_ROW], s=30, c='red', alpha=0.5, marker='3')
plt.scatter(y_test_reg, preds, s=30, c='green', alpha=0.5, marker='x')
plt.show()

model.save_model('classifier.model')
regressor.save_model('regressor.model')



cand_frame = pd.read_pickle("cand_frame_10k_20k")

model = XGBClassifier({'nthread':4}) #init model
model.load_model("C:/Users/felix/Documents/University/Thesis/BremsAdder_LHCb/classifier.model")
# calc_frame = return_calculation_frame(model, cand_frame, df)
# calc_frame.columns.tolist()
calc_frame.to_pickle("calc_frame_10k_20k")
