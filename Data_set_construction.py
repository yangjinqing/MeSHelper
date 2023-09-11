# -*- coding: UTF-8 -*-
import pandas as pd

def construct_negative_samples():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_remove/mesh_remove_negatives.txt', 'a') as ff:
        for year in range(2002, 2017):
            revision = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_remove/mesh_R(remove)_C(change).txt', sep='\t', header=None)
            revision.columns = ['DescriptorName', 'label', 'year']
            revision = revision[revision['label'] == 'R'].loc[:, ['DescriptorName', 'year']]
            tree = pd.read_csv('/home/lab109/data/MeSH_2001_2020/tree/mtrees' + str(year - 1) + '.bin', sep=';', header=None)
            tree.columns = ['DescriptorName', 'tree']

            # if one descriptor has two trees and both move, we only consider the first record.
            revision_year = revision[revision['year'] == year]['DescriptorName'].drop_duplicates(keep='first')
            # remove the moved descriptors and save the non-moves.
            non_revisions = set(tree['DescriptorName']) - set(revision_year)
            for non_revision in non_revisions:
                print(non_revision, year)
                ff.write(str(non_revision) + '|' + str(year) + '\n')

def gain_negative_samples():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_remove/mesh_remove_negative_samples.txt', 'a') as ff:
        negatives = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_remove/mesh_remove_negatives.txt', sep='|', header=None)
        negatives.columns = ['DescriptorName', 'year']
        revision = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_remove/mesh_R(remove)_C(change).txt', sep='\t', header=None)
        revision.columns = ['DescriptorName', 'label', 'year']
        revision=revision[revision['label'] == 'R'].loc[:, ['DescriptorName', 'year']]
        # print(negatives)
        for year in range(2002, 2017):
            # according the count of positive samples in each year, we extract the negative samples.
            n = revision[revision['year'] == year]['DescriptorName'].count()
            negative_samples = negatives[negatives['year'] == year]['DescriptorName'].sample(n=n, replace=False)
            for negative_sample in negative_samples:
                print(negative_sample, year)
                ff.write(str(negative_sample) + '|' + str(year) + '\n')

def construct_experiment_data():
    with open('/home/lab109/data/MeSH_2001_2020/revise/mesh_extension_negative_label.txt', 'a') as ff:
        for year in range(2002, 2017):
            extents = pd.read_csv('/home/lab109/data/MeSH_2001_2020/revise/mesh_extension_negative_samples.txt', sep='|', header=None)
            extents.columns = ['extension', 'year']
            tree = pd.read_csv('/home/lab109/data/MeSH_2001_2020/revise/mesh_depth_tre_negatives.txt', sep='|', header=None)
            tree.columns = ['extension', 'max_dep', 'min_dep', 'year', 'label']
            for extent in extents[extents['year'] == year]['extension'].drop_duplicates(keep='first').values:
                temp = tree[tree['year'] == year].loc[:, ['extension', 'label']]
                label = temp[temp['extension'] == extent].drop_duplicates(keep='first')['label'].values[0]
                print(extent, label, year)
                ff.write(str(extent) + '|' + str(label) + '|' + str(year) + '\n')
    ff.close()

def mesh_evolution_data():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_mesh.txt', 'a') as ff:
        move = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_move.txt', sep='\t', header=None)
        extension = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_new_father.txt', sep='|', header=None)
        RC = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_R(remove)_C(change).txt', sep='|', header=None)
        RC.columns = ['DescriptorName', 'label', 'year']
        Remove_C = RC.loc[:, ['DescriptorName', 'year', 'label']]
        move['label'] = 'M'
        move.columns = ['DescriptorName', 'year', 'label']
        extension['label'] = 'E'
        extension.columns = ['DescriptorName', 'new_mesh', 'year', 'label']
        extension_ = extension.loc[:, ['DescriptorName', 'year', 'label']]
        frame = [extension_, move, Remove_C]
        evolution = pd.concat(frame).sort_values(by='year')
    return evolution

# remove the revised descriptors for each year.
def construct_evolution_negative():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_negatives_tre.txt', 'a') as ff:
        for year in range(2002, 2017):
            revision = mesh_evolution_data()
            revision = revision.loc[:, ['DescriptorName', 'year']]
            # before evolution, we have the fixed position
            tree = pd.read_csv('/home/lab109/data/MeSH_2001_2020/tree/mtrees' + str(year - 1) + '.bin', sep=';', header=None)
            tree.columns = ['DescriptorName', 'tree']
            # print(tree)
            # if one descriptor has two trees and both evolute, we only consider the first record.
            revisions = revision[revision['year'] == year]['DescriptorName'].drop_duplicates(keep='first')
            # # remove the evolution descriptors and save the non-moves.
            non_revisions = set(tree['DescriptorName']) - set(revisions)
            for non_revision in non_revisions:
                tre = tree[tree['DescriptorName'] == non_revision]['tree'].values[0]
                print(non_revision, year, tre[0])
                ff.write(str(non_revision) + '|' + str(year) + '|' + str(tre[0]) + '\n')

#According to the tree branches and year, match the negative samples.
def construct_train_test_data():
    for i in range(2, 3):
        with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_negative_samples_'+str(i)+'.txt', 'a') as ff:
            evolution = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_mesh.txt', sep='|', header=None)
            evolution.columns = ['DescriptorName', 'year', 'label', 'tre']
            evolution['id'] = evolution['DescriptorName']+'|'+evolution['year'].astype(str)
            positives = evolution[evolution['year'] >= 2004]
            print(positives)
            non_evolution = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_negatives_tre.txt', sep='|', header=None)
            non_evolution.columns = ['DescriptorName', 'year', 'tre']
            non_evolution['id'] = non_evolution['DescriptorName'] + '|' + non_evolution['year'].astype(str)
            negatives = non_evolution[non_evolution['year'] >= 2004]
            # print(negatives.groupby(by='tre').size())
            for tre in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'V', 'Z']:
                # according the count of positive samples in each year, we extract the negative samples.
                n = positives[positives['tre'] == tre]['id'].count()
                print(n)
                negative_samples = negatives[negatives['tre'] == tre].loc[:, ['id', 'DescriptorName', 'year']].sample(n=n, replace=False)
                for id in negative_samples['id']:
                    DescriptorName = negative_samples[negative_samples['id'] == id]['DescriptorName'].values[0]
                    year = negative_samples[negative_samples['id'] == id]['year'].values[0]
                    print(str(id)+'|'+str(DescriptorName)+'|'+str(year)+'|'+str(tre))
                    ff.write(str(DescriptorName)+'|'+str(year)+'|'+str(tre)+'\n')

if __name__ == '__main__':
    construct_train_test_data()