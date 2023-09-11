from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

'''
Logiest function
'''
def Logiest(train, test, feature):
    print("Logiest mode")
    X_train = train[feature]
    X_test = test[feature]
    y_train = train['re']
    y_test  = test['re']
    param_grid = {'penalty': ['l1', 'l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100,120,150, 200]}
    print("Parameters:{}".format(param_grid))

    grid_search = RandomizedSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
    print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))

'''
Support Vector Machine Functions
'''
def SVM(train, test, feature):
    print("SVM model")
    X_train = train[feature]
    X_test = test[feature]
    y_train = train['re']
    y_test = test['re']
    # 把要调整的参数以及其候选值 列出来；
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100],"kernel":['linear', 'poly', 'rbf']}

    print("Parameters:{}".format(param_grid))

    grid_search = RandomizedSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)  # 实例化一个GridSearchCV类
    grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
    print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))

'''
Decision Trees Function
'''
def dt_classifier(train, test, feature):
    print("Decision Tree model")
    X_train = train[feature]
    X_test = test[feature]
    y_train = train['re']
    y_test = test['re']
    # 把要调整的参数以及其候选值 列出来；
    param_grid = {
        'criterion':['gini','entropy'],
        'splitter':['best','random'],
        'min_samples_leaf':range(1,100,5),
        'max_depth':range(1,100,5)
    }
    print("Parameters:{}".format(param_grid))
    grid_search = RandomizedSearchCV(DecisionTreeClassifier(), param_grid, cv=5, n_jobs=-1)  # 实例化一个GridSearchCV类
    grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
    print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))

'''
Random Forests Function
'''
def randomforest_classifier(train, test, feature):
    print("Randomforest_classifier model")
    X_train = train[feature]
    X_test = test[feature]
    y_train = train['re']
    y_test = test['re']
    # 把要调整的参数以及其候选值 列出来；
    param_grid = {
        'n_estimators': range(10, 1000, 10),
        'max_depth': range(1, 100, 1),
        'min_samples_leaf':range(1, 20, 1)
        # 'min_samples_split':range(1, 20, 1)
    }
    print("Parameters:{}".format(param_grid))
    grid_search = RandomizedSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)  # 实例化一个GridSearchCV类
    grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
    print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))

'''
GradientBoosting Function
'''
def GBoostingClassifier(train, test, feature):
    print("GradientBoostingClassifier model")
    X_train = train[feature]
    X_test = test[feature]
    y_train = train['re']
    y_test = test['re']
    # 把要调整的参数以及其候选值 列出来；
    param_grid = {
        'n_estimators': range(10, 1000, 2),
        'max_depth': range(1, 100, 1)
    }
    print("Parameters:{}".format(param_grid))
    grid_search = RandomizedSearchCV(GradientBoostingClassifier(), param_grid, cv=5, n_jobs=-1)  # 实例化一个GridSearchCV类
    grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
    print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))

'''
eXtreme Gradient Boosting Function
'''
def xgboost_classifier(train, test, feature):
    print("xgboost_classifier model")
    X_train = train[feature]
    X_test = test[feature]
    y_train = train['re']
    y_test = test['re']
    # 把要调整的参数以及其候选值 列出来
    param_grid = {
        'n_estimators': range(10, 1000, 10),
        'max_depth': range(1, 100, 1),
        'min_child_weight': range(1, 20, 1)
    }
    print("Parameters:{}".format(param_grid))
    grid_search = RandomizedSearchCV(XGBClassifier(), param_grid, cv=5, n_jobs=-1)  # 实例化一个GridSearchCV类
    grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
    print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))

'''
K-Nearest Neighbor Function
'''
def KNN_classifier(train, test, feature):
    print("KNeighborsClassifier model")
    X_train = train[feature]
    X_test = test[feature]
    y_train = train['re']
    y_test = test['re']
    # 把要调整的参数以及其候选值 列出来；
    param_grid = {
        'n_neighbors': range(1, 50, 1),
        'leaf_size': range(1, 50, 1),
        'weights': ['uniform','distance']
    }
    print("Parameters:{}".format(param_grid))
    grid_search = RandomizedSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1)  # 实例化一个GridSearchCV类
    grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
    print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))

'''
Multilayer Perceptron Function
'''
def mlp_classifier(train, test, feature):
    print("mlp_classifier model")
    X_train = train[feature]
    X_test = test[feature]
    y_train = train['re']
    y_test = test['re']
    param_grid = {
        'hidden_layer_sizes': range(10, 1000, 10),
        'max_iter': range(10, 500, 10),
        'activation':['identity', 'logistic', 'tanh', 'relu']
    }
    print("Parameters:{}".format(param_grid))
    grid_search = RandomizedSearchCV(MLPClassifier(), param_grid, cv=5, n_jobs=-1)  # 实例化一个GridSearchCV类
    grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
    print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))

if __name__ == '__main__':
    pass
