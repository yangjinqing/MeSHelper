import xgboost
# import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier


def test():
    # train XGBoost model
    import xgboost
    # import shap

    # train XGBoost model
    X, y = shap.datasets.adult()
    model = xgboost.XGBClassifier().fit(X, y)

    # compute SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)


def explain_RFs(train, feature):
    y=train['re']
    X = train[feature]
    # train an XGBoost model
    model = RandomForestClassifier(n_estimators=380, max_depth=13, min_samples_leaf=18).fit(X, y)

    # explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print(shap_values)
    fig = plt.gcf()
    fig.set_figheight(7)
    fig.set_figwidth(12)
    plt.rcParams['font.size'] = '20'
    shap.plots.scatter(shap_values[:,"count_sib"])
    plt.rcParams['font.size'] = 20
    plt.show()


def explain_RFs_for_single(train, feature):
    y=train['re']
    X = train[feature]
    # train an XGBoost model
    model = RandomForestClassifier(n_estimators=380, max_depth=13, min_samples_leaf=18)
    model.fit(X, y)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X, check_additivity=False)

    shap.plots.beeswarm(shap_values[0])
)

def explain_multi_RFs(train_x, tran_y):

    forest = RandomForestClassifier(random_state=0, n_estimators=901, max_depth=11)
    model = forest.fit(train_x, tran_y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_x)

    fig = plt.gcf()
    fig.set_figheight(7)
    fig.set_figwidth(12)
    plt.rcParams['font.size'] = '20'
    # plt.title(label='Extension')
    shap.dependence_plot("pub_count", shap_values[1], train_x, interaction_index='pub_count', show=True)
    # plt.title(label='Extension')
    shap.dependence_plot("count_dv", shap_values[1], train_x, interaction_index='count_dv', show=True)
    # plt.title(label='Extension')
    shap.dependence_plot("count_sta", shap_values[1], train_x, interaction_index='count_sta', show=True)

    shap.dependence_plot("citation_pub", shap_values[1], train_x, interaction_index='citation_pub', show=True)
    plt.rcParams['font.size'] = 20
    plt.tight_layout()
    plt.show()

def AUC_curve(fprs, tprs, roc_aucs):
    # plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 7))
    for i in range(0,6):
        name=['SVM', 'KNN', 'DTs', 'RFs', 'XGBoost', 'MLPs']
        fpr = fprs[i]
        tpr = tprs[i]
        roc_auc = roc_aucs[i]*100
        ax.plot(fpr, tpr, label=f"{name[i]} (AUC = {roc_auc:.2f})",lw=2)
        ax.grid(True, linestyle='-.')
    plt.xlabel('FPR', fontsize='25', color='0.2', fontweight='bold')
    plt.ylabel('TPR', fontsize='25', color='0.2', fontweight='bold')
    plt.xticks(fontsize='25')
    plt.yticks(fontsize='25')
    plt.legend(loc='best', fontsize='20')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test()
    # pass
    # explain_RFs()
