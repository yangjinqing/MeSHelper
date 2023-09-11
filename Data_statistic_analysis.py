import pandas as pd
import matplotlib.pyplot as plt

def revision_type_distribution():
    # move
    move = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_move.txt',sep='\t', header=None)
    move.columns=['DescriptorName','year']
    # print(move.groupby(['year']).count().values)
    move_value=[]
    for i in move.groupby(['year']).count().values:
        move_value.append(i[0])
    print(move_value[0:16])

    # new mesh fathers
    new_fathers = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_new_father.txt', sep='|', header=None)
    print(new_fathers[2])
    new_fathers_value=[]
    for j in new_fathers.groupby([2]).count().values:
        new_fathers_value.append(j[0])
    print(new_fathers_value)

    # changes
    changes = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_R(remove)_C(change).txt', sep='|', header=None)
    changes_value = []
    for j in changes.groupby([2]).count().values:
        changes_value.append(j[0])
    print(changes_value[0:16])

    year=[2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]

    # plt.style.use('ggplot')
    plt.subplots(1, 1, figsize=(12, 7))
    plt.plot(year, new_fathers_value, ls='-.', lw=2, color='green', label='Extension type')
    plt.plot(year, changes_value[0:16], ls=':', lw=2, color='purple', label='Change type')
    plt.plot(year, move_value[0:16], ls='-', lw=2, color='blue', label='Relocation type')
    plt.xlabel('Year', fontsize='25', fontweight='bold')
    plt.ylabel('MHs_evolution_number', fontsize='25', fontweight='bold')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    revision_type_distribution()