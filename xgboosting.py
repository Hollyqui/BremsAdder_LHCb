from itertools import compress
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
############# HELPER FUNCTIONS #######################


def train_classifier(training_data=None, X=None, y=None, silent=False,
                     scale_pos_weight=1, learning_rate=0.1, colsample_bytree = 0.4,
                     subsample = 0.8, objective='binary:logistic', n_estimators=100,
                     reg_alpha = 0.3, max_depth=7, gamma=1, early_stopping_rounds = 1000):
    # first get data
    if X is None or y is None:
        if training_data is None:
            try:
                training_data = pd.read_pickle(filename)
            except:
                print("Data neither passed along nor valid file location given")
        y = np.array(training_data['labels'])
        X = np.array(training_data.drop(['labels'], axis=1))
    # now train
    model = XGBClassifier(silent=silent,
                          scale_pos_weight=scale_pos_weight,
                          learning_rate=learning_rate,
                          colsample_bytree = colsample_bytree,
                          subsample = subsample,
                          objective='binary:logistic',
                          n_estimators=n_estimators,
                          reg_alpha = reg_alpha,
                          max_depth=max_depth,
                          gamma=gamma,
                          early_stopping_rounds = early_stopping_rounds)
    model.fit(X, y)
    return model

def train_regressor(training_data=None, X=None, y=None, silent=False,
                    scale_pos_weight=1, learning_rate=0.01, colsample_bytree = 0.4,
                    subsample = 0.8, n_estimators=1000, min_child_weight=0,
                    reg_alpha = 0.3, max_depth=6, gamma=1, early_stopping_rounds = 10000):
    # first get data
    if X is None or y is None:
        if training_data is None:
            try:
                training_data = pd.read_pickle(filename)
            except:
                print("Data neither passed along nor valid file location given")
        y = np.array(training_data['labels'])
        X = np.array(training_data.drop(['labels'], axis=1))
    # now train
    model = XGBRegressor(silent=silent,
                         scale_pos_weight=scale_pos_weight,
                         learning_rate=learning_rate,
                         colsample_bytree = colsample_bytree,
                         subsample = subsample,
                         n_estimators=n_estimators,
                         reg_alpha = reg_alpha,
                         min_child_weight = min_child_weight,
                         max_depth=max_depth,
                         gamma=gamma,
                         early_stopping_rounds = early_stopping_rounds)

    model.fit(X, y)
    return model



def plot_histograms(model, x_train, y_train):
    x_train = np.array(x_train, ndmin=2)
    y_train = np.array(y_train, ndmin=2)
    if (x_train.shape[0] != y_train.shape[0]):
        y_train = y_train.T
    if (x_train.shape[0] != y_train.shape[0]):
        print("x_train and y_train do not match in lenght: ", x_train.shape, " vs ", y_train.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=123)

    predictions = model.predict_proba(x_test)

    for i in range(0, int(np.max(y_train)+1)):
        class_var = np.array([predictions[j][i] for j in range(len(predictions))])
        df_test = pd.DataFrame()
        df_test["Net"] = class_var
        df_test["absid"] = y_test.T[0]


        crit_class1 = df_test['absid'] == i
        df_test_class1 = df_test[crit_class1]
        df_test_class2 = df_test - df_test[crit_class1]
        # log="y" transforms the scale to be logarithmic (in the following two lines)
        df_test_class1["Net"].plot.hist(bins=50, range=(0, 1), alpha=0.5, density=True, label="Class"+str(i))  # log="y")
        df_test_class2["Net"].plot.hist(bins=50, range=(0, 1), alpha=0.5, density=True, label=("Rest"))  # log="y")
        plt.legend(loc='upper right')
        plt.xlabel("ConvNet classifier")
        plt.show()


def get_metrics(model, X_test, y_test):
    preds = model.predict(X_test)
    best_preds = np.asarray([np.argmax(line) for line in preds])

    # print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
    # print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
    # print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))

    probs = model.predict_proba(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, np.argmax(model.predict_proba(X_test), axis=1)).ravel()
    print("True Positives", tp, "\nTrue Negatives:", tn,  "\nFalse Positives:", fp, "\nFalse Negatives", fn)
    print("Total Predictions:", tn+fp+fn+tp)
    plot_histograms(model, X_test, y_test)
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.show()
    # xgb.plot_importance(model)
################## XGBOOST MODEL #################################


# filename = "C:/Users/felix/Documents/University/Thesis/training_data"
# training_data = pd.read_pickle(filename)
# # training_data
# y = np.array(training_data['labels'])
# X = np.array(training_data.drop(['labels'], axis=1))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = training(x=X_train, y=y_train)
# get_metrics(model, X_test, y_test)


# plt.show()
