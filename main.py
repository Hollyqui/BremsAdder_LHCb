try:
    from Code.xgboosting import *
    from Code.data_preprocessor import *
except:
    from xgboosting import *
    from data_preprocessor import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
filename = "C:/Users/felix/Documents/University/Thesis/final_large_electron_set"
MAX_ROWS = 10000
df = pd.read_pickle(filename)
df = df[:MAX_ROWS]
# df = df.drop(['eminus_MCphotondaughters_ECAL_X'], axis=1)
# df = df.drop(['eminus_MCphotondaughters_ECAL_Y'], axis=1)
cand_frame = return_candidate_frame(df)
y = np.array(cand_frame['labels'])
X = np.array(cand_frame.drop(['labels'], axis=1))
X_train, X_test, y_train, y_test = train_test_split(X,y)
model = train_classifier(X=X_train, y=y_train)
get_metrics(model, X_test, y_test)

calc_frame = return_calculation_frame(model, cand_frame, df)
calc_frame


y_reg = np.array(calc_frame['labels'])
X_reg = np.array(calc_frame.drop(['labels'], axis=1))
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg,y_reg)
regressor = train_regressor(X=X_train_reg, y=y_train_reg)
preds = regressor.predict(X_test_reg)
plt.hist(y_test_reg-preds, bins=250)
plt.hist(y_test_reg, bins=250)

df.columns
# compare to BremsAdder_LHCb
P_REC = np.sqrt(df['eminus_BremPX']**2+df['eminus_BremPY']**2+df['eminus_BremPZ']**2)
