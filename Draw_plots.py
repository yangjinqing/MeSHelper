import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brokenaxes  import brokenaxes

def bar_databalance_plot():

    # labels = ['LogisRegression', 'Linear_SVC', 'K-Nearest Neighbor']
    EasyEnsemble_means_1 = [66.91]
    smote_enn_means_1 = [68.81]
    smote_means_1 = [61.91]

    EasyEnsemble_means_2 = [72.81]
    smote_enn_means_2 = [70.82]
    smote_means_2 = [72.09]

    EasyEnsemble_means_3 = [77.79]
    smote_enn_means_3 = [76.24]
    smote_means_3 = [78.11]

    EasyEnsemble_means_4 = [77.70]
    smote_enn_means_4 = [77.27]
    smote_means_4 = [79.01]

    EasyEnsemble_means_5 = [77.88]
    smote_enn_means_5 = [77.32]
    smote_means_5 = [77.27]

    EasyEnsemble_means_6 = [72.90]
    smote_enn_means_6 = [75.52]
    smote_means_6 = [74.87]

    x = np.arange(1)  # the label locations
    width = 1.20  # the width of the bars

    plt.rcParams['font.sans-serif'] = ['serif']
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    rects1 = ax[0, 0].bar(x-width, EasyEnsemble_means_1, label='EasyEnsemble')
    ax[0, 0].text(x-width, 66.91, 66.91, ha='center', va='bottom', fontsize=12)
    rects2 = ax[0, 0].bar(x, smote_enn_means_1, label='SMOTEENN')
    ax[0, 0].text(x, 68.81, 68.81, ha='center', va='bottom', fontsize=12)
    rects3 = ax[0, 0].bar(x+width, smote_means_1, label='SMOTE')
    ax[0, 0].text(x+width, 61.91, 61.91, ha='center', va='bottom', fontsize=12)


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0, 0].set_ylabel('F value * 100',fontsize=12)
    ax[0, 0].set_title('LinearSVC',fontsize=10, fontweight='bold')
    ax[0, 0].tick_params(axis='y', labelsize=12)
    ax[0, 0].legend(loc='lower left',fontsize=9)
    ax[0, 0].set_ylim(0, 100)


    ##
    rects1 = ax[0, 1].bar(x-width, EasyEnsemble_means_2, label='EasyEnsemble')
    rects2 = ax[0, 1].bar(x, smote_enn_means_2, label='SMOTEENN')
    rects3 = ax[0, 1].bar(x+width, smote_means_2, label='SMOTE')
    ax[0, 1].text(x - width, 72.81, 72.81, ha='center', va='bottom', fontsize=12)
    ax[0, 1].text(x, 70.82, 70.82, ha='center', va='bottom', fontsize=12)
    ax[0, 1].text(x + width, 72.09, 72.09, ha='center', va='bottom', fontsize=12)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0, 1].set_title('KNN',fontsize=10, fontweight='bold')
    # ax[0, 1].set_xlabel('Macro  Micro  Weighted')
    ax[0, 1].legend(loc='lower left', fontsize=9)


    ##
    rects1 = ax[0, 2].bar(x-width, EasyEnsemble_means_3, label='EasyEnsemble')
    rects2 = ax[0, 2].bar(x, smote_enn_means_3, label='SMOTEENN')
    rects3 = ax[0, 2].bar(x+width, smote_means_3,label='SMOTE')
    ax[0, 2].text(x - width, 77.79, 77.79, ha='center', va='bottom', fontsize=12)
    ax[0, 2].text(x, 78.11, 78.11, ha='center', va='bottom', fontsize=12)
    ax[0, 2].text(x + width, 76.24, 76.24, ha='center', va='bottom', fontsize=12)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0, 2].set_title('DTs',fontsize=10, fontweight='bold')
    ax[0, 2].legend(loc='lower left', fontsize=9)

    ##RandomForest
    rects1 = ax[1, 0].bar(x-width, EasyEnsemble_means_4, label='EasyEnsemble')
    rects2 = ax[1, 0].bar(x, smote_enn_means_4, label='SMOTEENN')
    rects3 = ax[1, 0].bar(x+width, smote_means_4, label='SMOTE')
    ax[1, 0].text(x - width, 77.70, 77.70, ha='center', va='bottom', fontsize=12)
    ax[1, 0].text(x, 77.27, 77.27, ha='center', va='bottom', fontsize=12)
    ax[1, 0].text(x + width, 79.01, 79.01, ha='center', va='bottom', fontsize=12)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[1, 0].set_ylabel('F value * 100', fontsize=12)
    ax[1, 0].set_title('RFs',fontsize=10, fontweight='bold')
    ax[1, 0].tick_params(axis='y', labelsize=12)
    ax[1, 0].legend(loc='lower left', fontsize=9)

    ##GradientBoosting
    rects1 = ax[1, 1].bar(x-width, EasyEnsemble_means_5, label='EasyEnsemble')
    rects2 = ax[1, 1].bar(x, smote_enn_means_5, label='SMOTEENN')
    rects3 = ax[1, 1].bar(x+width, smote_means_5, label='SMOTE')
    ax[1, 1].text(x - width, 77.88, 77.88, ha='center', va='bottom', fontsize=12)
    ax[1, 1].text(x, 77.27, 77.27, ha='center', va='bottom', fontsize=12)
    ax[1, 1].text(x + width, 77.32, 77.32, ha='center', va='bottom', fontsize=12)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[1, 1].set_title('XGBoost',fontsize=10, fontweight='bold')
    ax[1, 1].legend(loc='lower left', fontsize=9)

    ###MLP
    rects1 = ax[1, 2].bar(x-width, EasyEnsemble_means_6, label='EasyEnsemble')
    rects2 = ax[1, 2].bar(x, smote_enn_means_6,label='SMOTEENN')
    rects3 = ax[1, 2].bar(x+width, smote_means_6,label='SMOTE')
    ax[1, 2].text(x - width, 72.90, 72.90, ha='center', va='bottom', fontsize=12)
    ax[1, 2].text(x, 74.87, 74.87, ha='center', va='bottom', fontsize=12)
    ax[1, 2].text(x + width, 75.52, 75.52, ha='center', va='bottom', fontsize=12)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[1, 2].set_title('MLP', fontsize=10, fontweight='bold')
    ax[1, 2].legend(loc='lower left', fontsize=9)

    fig.tight_layout()
    plt.show()

def PubMed_distribution():
    data = pd.read_csv('/home/lab109/data/MeSH_2001_2020/PubMed_Timeline_Results_by_Year.csv')
    print(data)
    bax = brokenaxes(xlims=((1781, 1800),(1900, 2021)), despine=False)
    bax.plot(data['Year'], data['Count'], linewidth=2.5)
    bax.set_xlabel('Year', fontsize='15', fontweight='bold')
    bax.set_ylabel('Publication_frequency', fontsize='15', fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    PubMed_distribution()