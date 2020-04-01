import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

calc_frame = pd.read_pickle("calc_frame_10k_20k")

y_train = np.array(calc_frame['labels'])
X_train = np.array(calc_frame.drop(['labels'], axis=1))

calc_frame_test = pd.read_pickle("calc_frame_20k_30k")
y_test = np.array(calc_frame_test['labels'])
X_test = np.array(calc_frame_test.drop(['labels'], axis=1))


model = XGBClassifier(verbose=1)
model.fit(X_train, y_train)
print("test")
