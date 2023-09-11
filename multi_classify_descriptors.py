# -*- coding: UTF-8 --*--
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import roc_curve, auc

metric_names = ['acc', 'p', 'r', 'f1']

def calc_metrics(test_y, pred_y):
    prob = 0.5
    # pred_y_label = [1 if i > prob else 0 for i in pred_y]
    pred_y_label = pred_y
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

'''
Support Vector Machine Functions
'''
def Linear_SVC(train_x, tran_y, test_x):
    train_re = OneVsRestClassifier(LinearSVC(random_state=0, penalty='l2', C=70)).fit(train_x, tran_y)
    y_pred = train_re.predict(test_x)
    return y_pred

'''
K-Nearest Neighbor Function
'''
def KNN(train_x, tran_y, test_x):
    neigh = KNeighborsClassifier(n_neighbors=61, weights='uniform')
    multi_knn = OneVsRestClassifier(neigh)
    y_pred = multi_knn.fit(train_x, tran_y).predict(test_x)
    return y_pred

'''
Decision Trees Function
'''
def DecisionTree(train_x, tran_y, test_x):
    DT =DecisionTreeClassifier(random_state=0, max_depth=11, min_samples_leaf=91)
    multi_dt = OneVsRestClassifier(DT)
    y_pred = multi_dt.fit(train_x, tran_y).predict(test_x)
    return y_pred

'''
Random Forests Function
'''
def GradientBoosting(train_x, tran_y, test_x):
    GB = GradientBoostingClassifier(random_state=0)
    multi_gb = OneVsRestClassifier(GB)
    y_pred = multi_gb.fit(train_x, tran_y).predict(test_x)
    return y_pred

'''
eXtreme Gradient Boosting Function
'''
def XGBoosting(train_x, tran_y, test_x):
    XGB = XGBClassifier(random_state=0, n_estimators=901, max_depth=11)
    multi_xgb = OneVsRestClassifier(XGB)
    y_pred = multi_xgb.fit(train_x, tran_y).predict(test_x)
    return y_pred

'''
Random Forests Function
'''
def RandomForest(train_x, tran_y, test_x):
    forest = RandomForestClassifier(random_state=0, n_estimators=901, max_depth=11)
    multi_target_forest = OneVsRestClassifier(forest)
    y_pred = multi_target_forest.fit(train_x, tran_y).predict(test_x)
    # print(forest.feature_importances_)
    return y_pred

'''
Logiest function
'''
def LogisRegression(train_x, tran_y, test_x):
    lr = LogisticRegression(random_state=0)
    multi_lr = OneVsRestClassifier(lr)
    y_pred = multi_lr.fit(train_x, tran_y).predict(test_x)
    return y_pred

'''
Multilayer Perceptron Function
'''
def MLP_Classifier(train_x, tran_y, test_x):
    Mlp =MLPClassifier(random_state=0, hidden_layer_sizes=210, max_iter=60)
    multi_mlp = OneVsRestClassifier(Mlp)
    y_pred = multi_mlp.fit(train_x, tran_y).predict(test_x)
    return y_pred

'''
TabNet Function
'''
def TabNet_Classifier(train_x, tran_y, test_x):
    Tab =TabNetClassifier()
    Tab.fit(train_x, tran_y)
    y_pred =Tab.predict(test_x)
    return y_pred

def select_model(X_resampled, y_resampled, test_X, function):
    if function == 'LogisRegression':
        y_pred = LogisRegression(X_resampled, y_resampled, test_X)
        return y_pred

    elif function == 'Linear_SVC':
        y_pred = Linear_SVC(X_resampled, y_resampled, test_X)
        return y_pred

    elif function == 'KNN':
        y_pred = KNN(X_resampled, y_resampled, test_X)
        return y_pred

    elif function == 'DecisionTree':
        y_pred = DecisionTree(X_resampled, y_resampled, test_X)
        return y_pred

    elif function == 'RandomForest':
        y_pred = RandomForest(X_resampled, y_resampled, test_X)
        return y_pred

    elif function == 'XGBoosting':
        y_pred = XGBoosting(X_resampled, y_resampled, test_X)
        return y_pred

    elif function == 'MLP':
        y_pred = MLP_Classifier(X_resampled, y_resampled, test_X)
        return y_pred


def multi_classification(train_X, train_y, df_test, features):
    df_test = shuffle(df_test)
    df_test_y = df_test['re']
    print('..............................')
    print(df_test_y)
    test_X = df_test[features]
    test_y = np.array(df_test_y).astype('int')
    test_X = np.array(test_X)
    train_X = np.array(train_X)
    train_y = np.array(train_y).astype('int')


    for func in ['Linear_SVC','KNN','DecisionTree','RandomForest','XGBoosting','MLP']:
        y_pred = select_model(train_X,train_y,test_X, func)
        print(str(func)+'************************')
        print(f"f1_score_macro: "
              f"{f1_score(test_y, y_pred, average='macro')}")
        print(f"f1_score_micro: "
              f"{f1_score(test_y, y_pred, average='micro')}")
        print(f"f1_score_weighted: "
              f"{f1_score(test_y, y_pred, average='weighted')}")
        print(classification_report(test_y, y_pred, labels=[0, 1, 2]))
        print(classification_report_imbalanced(test_y, y_pred, target_names=['class0', 'class1', 'class2']))



if __name__ == '__main__':
    pass
