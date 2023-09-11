import logging as log
import time
import warnings
from enum import Enum
import xlrd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from xgboost import XGBClassifier
import Model_explainability as Me

warnings.filterwarnings('ignore')
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)

metric_names = ['acc', 'p', 'r', 'f1']  #

# Metrics for model evolution
def calc_metrics(test_y, pred_y, average='macro_f1', search_cut_off=True):
    prob = 0.5
    pred_y_label = [1 if i > prob else 0 for i in pred_y]

    acc = accuracy_score(test_y, pred_y_label)
    p = precision_score(test_y, pred_y_label)
    r = recall_score(test_y, pred_y_label)

    macro_f1 = f1_score(test_y, pred_y_label, average='macro')
    micro_f1 = f1_score(test_y, pred_y_label, average='micro')

    pos_label_f1 = f1_score(test_y, pred_y_label, average='weighted')
    return dict(
        zip(['acc', 'p', 'r', 'f1', 'macro_f1', 'micro_f1', 'decision_value'],
            [acc, p, r, pos_label_f1, macro_f1, micro_f1, prob]))


def pprint(kv: list, decimal=2, pctg=False, sep=None):
    k = [item[0] for item in kv]
    if pctg:
        v = [round(item[1] * 100.0, decimal) for item in kv]
    else:
        v = [round(item[1], decimal) for item in kv]
    if not sep:
        df = pd.DataFrame(data=[v], columns=k)
        print(df.head())
    else:
        print(sep.join([str(s) for s in v]))

class ModelName(Enum):
    linear = 'Linear'
    logistic = 'Logistic'
    dt = 'DecisionTree'
    svm = 'SVM'
    knn = 'KNN'
    xgboost = 'XGBoost'
    randomforest = 'RandomForest'
    gb = 'GradientBoosting'
    mlp = 'MultiLayerPerceptron'

    @classmethod
    def available_modes(self):
        return [self.svm, self.knn, self.dt, self.randomforest, self.xgboost, self.mlp]
        # return [self.dt, self.randomforest]  # , self.c45
        # return [self.linear, self.logistic, self.dt, self.randomforest] # self.svm,

    @classmethod
    def get_short_name(self, model_name):
        return \
            dict(zip(
                [ self.svm, self.knn, self.dt, self.randomforest, self.xgboost, self.mlp],
                ['SVM', 'KNN', 'DecisionTree', 'RF', 'XGB', 'MLP']))[model_name]  #

# print('dataset size before deduplication', df.shape)
mode_names = ModelName.available_modes()
print('available_modes: ', mode_names)

def get_positive_scores(y_pred):
    y_ps = []
    for y_p in y_pred:
        # print(y_p[1])
        y_ps.append(y_p[1])
    return y_ps, []

def linear_classifier(X_train, Y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
    return y_pred, model.coef_

'''
Logiest function
'''
def logistic_classifier(X_train, Y_train, X_test):
    model = LogisticRegression(max_iter=1000, tol=1e-4, class_weight='balanced', C=2)
    model.fit(X_train, Y_train)
    y_pred = model.predict_proba(X_test)
    y_pred = [p1 for (p0, p1) in y_pred]
    y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
    return y_pred, model.coef_, model.intercept_

'''
Decision Trees Function
'''

def dt_classifier(X_train, Y_train, X_test):
    model = DecisionTreeClassifier(criterion='gini',
                                   splitter='random',
                                   max_depth=46,
                                   min_samples_leaf=36)

    model.fit(X_train, Y_train)
    # y_pred = model.predict(X_test)
    y_pred = model.predict_proba(X_test)
    return get_positive_scores(y_pred), model.feature_importances_


def svm_classifier(X_train, Y_train, X_test):
    model = SVC()  # max_depth=,
    model.fit(X_train, Y_train)
    # y_pred = model.predict(X_test)
    y_pred = model.decision_function(X_test)
    return y_pred, []

'''
K-Nearest Neighbor Function
'''
def KNN_classifier(X_train, Y_train, X_test):
    model = KNeighborsClassifier(weights='uniform', n_neighbors=6, leaf_size=27)
    model.fit(X_train, Y_train)
    # y_pred = model.predict(X_test)
    y_pred = model.predict_proba(X_test)
    return get_positive_scores(y_pred), []

'''
eXtreme Gradient Boosting Function
'''
def xgboost_classifier(X_train, Y_train, X_test):
    model = XGBClassifier(n_estimators=840,
                          max_depth=2,
                          min_child_weight=3)
    # return model
    model.fit(X_train, Y_train)
    # y_pred = model.predict(X_test)
    y_pred = model.predict_proba(X_test)
    return get_positive_scores(y_pred), []

'''
Random Forests Function
'''
def randomforest_classifier(X_train, Y_train, X_test):
    model = RandomForestClassifier(n_estimators=380,
                                   max_depth=13,
                                   min_samples_leaf=18)
    model.fit(X_train, Y_train)
    # y_pred = model.predict(X_test)
    y_pred = model.predict_proba(X_test)
    # print(model.feature_importances_)
    return get_positive_scores(y_pred), []

'''
GradientBoosting Function
'''
def gb_classifier(X_train, Y_train, X_test):
    model = GradientBoostingClassifier()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    return y_pred, model.feature_importances_

'''
Multilayer Perceptron Function
'''
def mlp_classifier(X_train, Y_train, X_test):
    model = MLPClassifier(hidden_layer_sizes=(370,), max_iter=310,
                          activation='relu')
    model.fit(X_train, Y_train)
    # y_pred = model.predict(X_test)
    y_pred = model.predict_proba(X_test)
    return get_positive_scores(y_pred), []


def use_classifier(X_train, Y_train, X_test, model_switch: str):
    if model_switch == ModelName.linear:
        pred_y, feature_importance = linear_classifier(X_train, Y_train, X_test)
    elif model_switch == ModelName.logistic:
        pred_y, coefs, _ = logistic_classifier(X_train, Y_train, X_test)
        feature_importance = np.array(coefs[0])
    elif model_switch == ModelName.dt:
        pred_y, feature_importance = dt_classifier(X_train, Y_train, X_test)
    elif model_switch == ModelName.svm:
        pred_y, feature_importance = svm_classifier(X_train, Y_train, X_test)
    elif model_switch == ModelName.knn:
        pred_y, feature_importance = KNN_classifier(X_train, Y_train, X_test)
    elif model_switch == ModelName.xgboost:
        pred_y, feature_importance = xgboost_classifier(X_train, Y_train, X_test)
    elif model_switch == ModelName.randomforest:
        pred_y, feature_importance = randomforest_classifier(X_train, Y_train, X_test)
    elif model_switch == ModelName.gb:
        pred_y, feature_importance = gb_classifier(X_train, Y_train, X_test)
    elif model_switch == ModelName.mlp:
        pred_y, feature_importance = mlp_classifier(X_train, Y_train, X_test)
    else:
        pass
    return pred_y, feature_importance


def classify_descriptor(df_train, df_test, features):
    feature_names = features
    df_train = shuffle(df_train)
    df_test = shuffle(df_test)
    df_train_y = df_train['re']
    df_test_y = df_test['re']
    print(df_test_y.T)
    fprs =[]
    tprs =[]
    roc_aucs = []
    for idx, model_switch in enumerate(mode_names):
        train_X = df_train[features]
        train_y = np.array(df_train_y).astype('int')
        test_X = df_test[features]
        test_y = np.array(df_test_y).astype('int')

        print('-' * 160)
        print(str(model_switch) + '\tused features:\n', '\t'.join(feature_names))
        avg_metrics = []
        pred_y, feature_importance = use_classifier(train_X, train_y, test_X, model_switch=model_switch)
        print(pred_y)
        fpr, tpr, threshold = roc_curve(test_y, pred_y, pos_label=1)  ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)
    print(fprs)
    print(tprs)
    print(roc_aucs)
    Me.AUC_curve(fprs, tprs, roc_aucs)




if __name__ == '__main__':
    pass