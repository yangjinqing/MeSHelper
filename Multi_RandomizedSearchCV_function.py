# -*- coding: UTF-8 --*--
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from imblearn.metrics import classification_report_imbalanced
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


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
def Linear_SVC(train_x, tran_y, test_x,c_value, pen):
    train_re = OneVsRestClassifier(LinearSVC(multi_class='ovr', random_state=10, penalty=pen, C=c_value)).fit(train_x, tran_y)
    y_pred = train_re.predict(test_x)
    return y_pred, c_value, pen

'''
GradientBoosting Function
'''
def GradientBoosting(train_x, tran_y, test_x, n_est, max_dep):
    GB = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_dep, random_state=10)
    multi_gb = OneVsRestClassifier(GB)
    y_pred = multi_gb.fit(train_x, tran_y).predict(test_x)
    return y_pred, n_est, max_dep

'''
Decision Trees Function
'''
def DecisionTree(train_x, tran_y, test_x, max_dep, min_leaf ):
    DT =DecisionTreeClassifier(max_depth=max_dep,min_samples_leaf=min_leaf, random_state=0)
    multi_dt = OneVsRestClassifier(DT)
    y_pred = multi_dt.fit(train_x, tran_y).predict(test_x)
    return y_pred, max_dep, min_leaf

'''
K-Nearest Neighbor Function
'''
def KNN(train_x, tran_y, test_x, n_est, weight):
    neigh = KNeighborsClassifier(n_neighbors=n_est, weights=weight)
    multi_knn = OneVsRestClassifier(neigh)
    y_pred = multi_knn.fit(train_x, tran_y).predict(test_x)
    return y_pred, n_est, weight

'''
Random Forests Function
'''
def RandomForest(train_x, tran_y, test_x, n_est, max_dep):
    forest = RandomForestClassifier(n_estimators=n_est, max_depth=max_dep, random_state=10)
    multi_target_forest = OneVsRestClassifier(forest)
    y_pred = multi_target_forest.fit(train_x, tran_y).predict(test_x)
            # print(forest.feature_importances_)
    return y_pred, n_est, max_dep

'''
eXtreme Gradient Boosting Function
'''
def XGBClass(train_x, tran_y, test_x, n_est, max_dep):
    XGB = XGBClassifier(n_estimators=n_est, max_depth=max_dep,random_state=10, n_jobs=-1)
    multi_target_XGB = OneVsRestClassifier(XGB)
    y_pred = multi_target_XGB.fit(train_x, tran_y).predict(test_x)
            # print(forest.feature_importances_)
    return y_pred, n_est, max_dep

'''
Logiest function
'''
def LogisRegression(train_x, tran_y, test_x, c_value, pen):
    lr = LogisticRegression(multi_class="ovr", random_state=10,penalty=pen, C=c_value)
    multi_lr = OneVsRestClassifier(lr)
    y_pred = multi_lr.fit(train_x, tran_y).predict(test_x)
    return y_pred,c_value, pen


def select_model(X_resampled, y_resampled, test_X, function, para1, para2):
    if function == 'LogisRegression':
        y_pred, c_value, pen = LogisRegression(X_resampled, y_resampled, test_X, para1, para2)
        return y_pred, c_value, pen

    elif function == 'Linear_SVC':
        y_pred,c_value,pen = Linear_SVC(X_resampled, y_resampled, test_X, para1, para2)
        return y_pred, c_value, pen

    elif function == 'KNN':
        y_pred, n_est, leaf = KNN(X_resampled, y_resampled, test_X, para1, para2)
        return y_pred, n_est, leaf

    elif function == 'DecisionTree':
        y_pred, max_dep, min_leaf = DecisionTree(X_resampled, y_resampled, test_X, para1, para2)
        return y_pred, max_dep, min_leaf

    elif function == 'RandomForest':
        y_pred, n_est, max_dep = RandomForest(X_resampled, y_resampled, test_X, para1, para2)
        return y_pred, n_est, max_dep

    elif function == 'GradientBoosting':
        y_pred, n_est, max_dep = GradientBoosting(X_resampled, y_resampled, test_X, para1, para2)
        return y_pred, n_est, max_dep

    elif function == 'XGBClass':
        y_pred, n_est, max_dep = XGBClass(X_resampled, y_resampled, test_X, para1, para2)
        return y_pred, n_est, max_dep



def multi_classification(X,Y, function, para1, para2):

    # df_train = shuffle(df_train)
    # df_test = shuffle(df_test)
    # # df_train_y = df_train['re']
    # df_test_y = df_test['label']
    # # train_X = df_train[features]
    # # train_y = np.array(df_train_y)
    # # print('train_y')
    # # print(list(train_y))
    # test_X = df_test[features]
    # test_y = np.array(df_test_y)
    # print('******')
    # print(list(df_test_y))
    # print('test_y')
    # print(list(test_y))
    Y = np.array(Y).astype('int')
    # print(Y)
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X = np.array(X_scaled)
    kf = KFold(n_splits=5, shuffle=False)
    index_split = kf.split(Y)
    # print(indx_split)
    avg_metrics_macro = []
    avg_metrics_micro = []
    avg_metrics_weighted = []
    for train_index, test_index in index_split:
        # print('******************************************')
        # print(len(train_index))
        # print(len(test_index))
        train_X, train_y = X[train_index], Y[train_index]
        test_X, test_y = X[test_index], Y[test_index]
        # print(test_X)
        rus = RandomUnderSampler(random_state=0)
        smote_enn = SMOTEENN(random_state=0)
        smote = SMOTE(random_state=0)
        X_resampled, y_resampled = smote.fit_resample(train_X, train_y)
        # print("############################################")
        # print(len(X_resampled))
        # print(len(y_resampled))
        # print(sorted(Counter(y_resampled).items()))
        # y=pd.DataFrame(y_resampled)
        # print(y[y[0]==2].count())

        y_pred, para1, para2 = select_model(X_resampled, y_resampled, test_X, function, para1, para2)

        # print(classification_report(test_y, y_pred, labels=[0, 1, 2]))
        # print(classification_report_imbalanced(test_y, y_pred, target_names=['class1', 'class2', 'class3']))
        # print(str(function)+'*********************************F1_value')
        avg_metrics_macro.append(f1_score(test_y, y_pred, average='macro')*100)
        # avg_metrics_micro.append(f1_score(test_y, y_pred, average='micro')*100)
        # avg_metrics_weighted.append(f1_score(test_y, y_pred, average='weighted')*100)

    # print(str(function)+'*********************************F1_values')
    print(str(para1)+'\t'+str(para2)+'\t'+str(np.mean(avg_metrics_macro)))
    # print(f"avg_metrics_macro:"  f"{np.mean(avg_metrics_macro)}"   f"{np.var(avg_metrics_macro)}")
    # print(f"avg_metrics_micro:"  f"{np.mean(avg_metrics_micro)}"    f"{np.var(avg_metrics_micro)}")
    # print(f"avg_metrics_weighted:" f"{np.mean(avg_metrics_weighted)}" f"{np.var(np.mean(avg_metrics_weighted))}")






if __name__ == '__main__':
    pass
    # X, y = datasets.load_iris(return_X_y=True)
    # train_x = X[21:]
    # train_y = y[21:]
    # test_x = X[1:20]
    # test_y = y[1:20]
    # multi_classification(train_x, train_y, test_x, test_y)