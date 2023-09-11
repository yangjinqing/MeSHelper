# -*- coding: UTF-8 --*--
import pandas as pd
import numpy as np
from imblearn.ensemble import EasyEnsembleClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle

metric_names = ['acc', 'p', 'r', 'f1']
def calc_metrics(test_y, pred_y):
    prob = 0.5
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
def Linear_SVC(train_x, tran_y, test_x, para1, para2):
    train_re = OneVsRestClassifier(LinearSVC(random_state=0, C=para1, penalty=para2)).fit(train_x, tran_y)
    model_gb = EasyEnsembleClassifier(base_estimator=train_re)
    y_pred = model_gb.fit(train_x, tran_y).predict(test_x)
    return y_pred

'''
K-Nearest Neighbor Function
'''
def KNN(train_x, tran_y, test_x, para1, para2):
    neigh = KNeighborsClassifier(n_neighbors=para1, weights=para2)
    multi_knn = OneVsRestClassifier(neigh)
    model_knn = EasyEnsembleClassifier(base_estimator=multi_knn)
    y_pred = model_knn.fit(train_x, tran_y).predict(test_x)
    return y_pred

'''
Decision Trees Function
'''
def DecisionTree(train_x, tran_y, test_x, para1, para2):
    DT =DecisionTreeClassifier(random_state=0, max_depth=para1, min_samples_leaf=para2)
    multi_dt = OneVsRestClassifier(DT)
    model_dt = EasyEnsembleClassifier(base_estimator=multi_dt)
    y_pred = model_dt.fit(train_x, tran_y).predict(test_x)
    return y_pred

'''
GradientBoosting Function
'''
def GradientBoosting(train_x, tran_y, test_x):
    GB = GradientBoostingClassifier(random_state=0)
    multi_gb = OneVsRestClassifier(GB)
    y_pred = multi_gb.fit(train_x, tran_y).predict(test_x)
    return y_pred

'''
eXtreme Gradient Boosting Function
'''
def XGBoosting(train_x, tran_y, test_x, para1, para2):
    XGB = XGBClassifier(random_state=0, n_estimators=para1, max_depth=para2)
    multi_target_xgb = OneVsRestClassifier(XGB)
    multi_xgb = EasyEnsembleClassifier(base_estimator=multi_target_xgb)
    y_pred = multi_xgb.fit(train_x, tran_y).predict(test_x)
    return y_pred

'''
Random Forests Function
'''
def RandomForest(train_x, tran_y, test_x, para1, para2):
    forest = RandomForestClassifier(random_state=0, n_estimators=para1, max_depth=para2)
    multi_target_forest = OneVsRestClassifier(forest)
    model_forest = EasyEnsembleClassifier(base_estimator=multi_target_forest)
    y_pred = model_forest.fit(train_x, tran_y).predict(test_x)
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
def MLP_Classifier(train_x, tran_y, test_x, para1, para2):
    Mlp =MLPClassifier(random_state=0, hidden_layer_sizes=para1, max_iter=para2)
    multi_target_mlp = OneVsRestClassifier(Mlp)
    model_mlp = EasyEnsembleClassifier(base_estimator=multi_target_mlp)
    y_pred = model_mlp.fit(train_x, tran_y).predict(test_x)
    return y_pred

'''
Tabnet Function
'''
def TabNet_Classifier(train_x, tran_y, test_x):
    Tab =TabNetClassifier()
    Tab.fit(train_x, tran_y)
    y_pred =Tab.predict(test_x)
    return y_pred

def select_model(X_resampled, y_resampled, test_X,para1, para2, function):
    if function == 'LogisRegression':
        y_pred = LogisRegression(X_resampled, y_resampled, test_X)
        return y_pred

    elif function == 'Linear_SVC':
        y_pred = Linear_SVC(X_resampled, y_resampled, test_X, para1, para2)
        return y_pred

    elif function == 'KNN':
        y_pred = KNN(X_resampled, y_resampled, test_X, para1, para2)
        return y_pred

    elif function == 'DecisionTree':
        y_pred = DecisionTree(X_resampled, y_resampled, test_X, para1, para2)
        return y_pred

    elif function == 'RandomForest':
        y_pred = RandomForest(X_resampled, y_resampled, test_X, para1, para2)
        return y_pred

    elif function == 'XGBoosting':
        y_pred = XGBoosting(X_resampled, y_resampled, test_X, para1, para2)
        return y_pred

    elif function == 'MLP':
        y_pred = MLP_Classifier(X_resampled, y_resampled, test_X,para1, para2)
        return y_pred


def multi_classification(train_X, train_y, df_test, features, para1, para2, func):
    # df_train = shuffle(df_train)
    df_test = shuffle(df_test)
    # df_train_y = df_train['re']
    df_test_y = df_test['re']
    # print('..............................')
    test_X = df_test[features]
    test_y = np.array(df_test_y).astype('int')
    test_X = np.array(test_X)
    train_X = np.array(train_X)
    train_y = np.array(train_y).astype('int')
    y_pred = select_model(train_X, train_y, test_X,para1, para2, func)
    # print(str(func) + '************************')
    print(f"f1_score_macro: "
          f"{f1_score(test_y, y_pred, average='macro')}")
    with open('/home/lab109/data/MeSH_2001_2020/'+str(func)+'.txt','a') as f:
        f.write(str(f1_score(test_y, y_pred, average='macro'))+'\t'+str(para1)+'\t'+str(para2)+'\n')

if __name__ == '__main__':
    pass