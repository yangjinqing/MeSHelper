# -*- coding: UTF-8 -*-
import igraph as g
import pandas as pd
from scipy.optimize import leastsq
import numpy as np
from clickhouse_driver import Client as click_client
import networkx as nx
import re
import igraph as ig
import threading

def click_server(sql):
    chs_host = 'localhost'
    chs_user = 'default'
    chs_pwd = 'root'
    chs_port = '9001'
    chs_database = 'pubmed20'
    client = click_client(host=chs_host, port=chs_port, user=chs_user, password=chs_pwd, database=chs_database,
                          send_receive_timeout=5)

    data, columns = client.execute(
        sql, columnar=True, with_column_types=True)
    df = pd.DataFrame({re.sub(r'\W', '_', col[0]): d for d, col in zip(data, columns)})
    return df


def get_positive():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_positive.txt', 'a') as ff:
        for year in range(2002, 2017):
            evolution = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_mesh.txt', sep='|', header= None)
            evolution.columns = ['DescriptorName', 'year', 'label', 'tre']
            for DescriptorName in evolution[evolution['year'] == year]['DescriptorName']:
                tre = evolution[evolution['DescriptorName'] == DescriptorName]['tre'].values[0]
                print(DescriptorName, year, tre)
                ff.write(str(DescriptorName)+'|'+str(year)+'|'+str(tre)+'\n')


def temporal_classifier_children_name():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_negative_children_name.txt', 'a') as ff:
        for year in range(2004, 2017):
            evolution = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_negative_samples_1.txt', sep='|', header=None)
            evolution.columns = ['DescriptorName', 'year', 'tre']
            SN = create_semantic_network(year-1)
            for DescriptorName in evolution[evolution['year'] == year]['DescriptorName'].drop_duplicates(keep='first'):
                # get the max value of paths and out direct
                if DescriptorName in SN.vs['name']:
                    children = SN.get_all_simple_paths(DescriptorName, cutoff=-1, mode='out')
                    pairs = []
                    vec = []
                    for child in children:
                        vec.append(child[-1])
                        if len(child) == 2:
                            pairs.append(child[-1])
                        else:
                            pass
                    vec_df = pd.DataFrame(vec).drop_duplicates(keep='first')
                    pairs_df = pd.DataFrame(pairs).drop_duplicates(keep='first')
                    # print(vec_df)
                    for av in vec_df.values:
                        print(DescriptorName, SN.vs[av[0]]['name'], year, 'av')
                        ff.write(str(DescriptorName) + '|' + str(SN.vs[av[0]]['name']) + '|' + str(year) + '|' + 'av' + '\n')
                    for dv in pairs_df.values:
                        print(str(DescriptorName) + '|' + str(SN.vs[dv[0]]['name']) + '|' + str(year) + '|' + 'dv')
                        ff.write(str(DescriptorName) + '|' + str(SN.vs[dv[0]]['name']) + '|' + str(year) + '|' + 'dv' + '\n')


def create_semantic_network(year):
    df = pd.read_csv('/home/lab109/data/MeSH_2001_2020/tree/mesh_net' + str(year) + '.txt', sep='|', header=None)
    df.columns = ['tree_start', 'tree_end', 'start', 'end', 'label']
    edges = df[df['label'] == 'f-s'].loc[:, ['start', 'end']]
    SN = g.Graph.DataFrame(edges, directed=True)
    return SN

# calculate the siblings of evolution
def temporal_classifier_siblings():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_siblings_positive.txt', 'a') as ff:
        for year in range(2004, 2017):
            evolution = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_positive.txt', sep='|', header=None)
            evolution.columns = ['DescriptorName', 'year', 'tre']
            tree = pd.read_csv('/home/lab109/data/MeSH_2001_2020/tree/mesh_net'+str(year-1)+'.txt', sep='|', header= None)
            tree.columns = ['tree_start', 'tree_end', 'start', 'end', 'label']
            siblings = tree[tree['label'] == 'b-b']
            # print(siblings)
            for DescriptorName in evolution[evolution['year'] == year]['DescriptorName'].drop_duplicates(keep='first'):
                trees_end = siblings[siblings['end'] == DescriptorName]['start']
                trees_start = siblings[siblings['start'] == DescriptorName]['end']
                count = trees_end.drop_duplicates(keep='first').count() + trees_start.drop_duplicates(keep='first').count()
                print(DescriptorName + '|' + str(count) + '|' + str(year))
                ff.write(DescriptorName + '|' + str(count) + '|' + str(year) + '\n')


# calculate the siblings of evolution for negatives
def temporal_classifier_siblings_negatives():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_siblings_negative_1.txt', 'a') as ff:
        for year in range(2004, 2017):
            non_evolution = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_negative_samples_1.txt', sep='|', header=None)
            non_evolution.columns = ['DescriptorName', 'year', 'tre']
            tree = pd.read_csv('/home/lab109/data/MeSH_2001_2020/tree/mesh_net' + str(year-1) + '.txt', sep='|')
            tree.columns = ['tree_start', 'tree_end', 'start', 'end', 'label']
            siblings = tree[tree['label'] == 'b-b']
            # print(siblings)
            for DescriptorName in non_evolution[non_evolution['year'] == year]['DescriptorName'].drop_duplicates(keep='first'):
                trees_end = siblings[siblings['end'] == DescriptorName]['start']
                trees_start = siblings[siblings['start'] == DescriptorName]['end']
                count = trees_end.drop_duplicates(keep='first').count() + trees_start.drop_duplicates(keep='first').count()
                print(DescriptorName + '|' + str(count) + '|' + str(year))
                ff.write(DescriptorName + '|' + str(count) + '|' + str(year) + '\n')

def temporal_classifier_children():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_children_positive.txt', 'a') as ff:
        for year in range(2004, 2017):
            evolution = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_positive.txt', sep='|', header=None)
            evolution.columns = ['DescriptorName', 'year', 'tre']
            SN = create_semantic_network(year-1)
            # print()
            for DescriptorName in evolution[evolution['year'] == year]['DescriptorName'].drop_duplicates(keep='first'):
                # get the max value of paths and out direct
                if DescriptorName in SN.vs['name']:
                    children = SN.get_all_simple_paths(DescriptorName, mode='out')
                    pairs = []
                    vec = []
                    for child in children:
                        vec.append(child[-1])
                        if len(child) == 2:
                            pairs.append(child[-1])
                        else:
                            pass
                    vec_df = pd.DataFrame(vec)
                    pairs_df = pd.DataFrame(pairs)
                    vec_count = vec_df.drop_duplicates(keep='first').count().values
                    pairs_count = pairs_df.drop_duplicates(keep='first').count().values
                    if len(vec_count) == 0:
                        vec_number = 0
                    else:
                        vec_number = vec_count[0]
                    if len(pairs_count) == 0:
                        pair_number = 0
                    else:
                        pair_number = pairs_count[0]
                    print(pairs_df)
                    print(str(pair_number) + '###########')
                    print(vec_df)
                    print(str(vec_number) + '***************')
                    ff.write(DescriptorName + '|' + str(pair_number) + '|' + str(vec_number) + '|' + str(year) + '\n')

def temporal_classifier_children_negatives():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_children_negative_1.txt', 'a') as ff:
        for year in range(2004, 2017):
            non_evolution = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_negative_samples_1.txt', sep='|')
            non_evolution.columns = ['DescriptorName', 'year', 'tre']
            SN = create_semantic_network(year-1)
            for DescriptorName in non_evolution[non_evolution['year'] == year]['DescriptorName'].drop_duplicates(keep='first'):
                # get the max value of paths and out direct
                if DescriptorName in SN.vs['name']:
                    children = SN.get_all_simple_paths(DescriptorName, mode='out')
                    pairs = []
                    vec = []
                    for child in children:
                        vec.append(child[-1])
                        if len(child) == 2:
                            pairs.append(child[-1])
                        else:
                            pass
                    vec_df = pd.DataFrame(vec)
                    pairs_df = pd.DataFrame(pairs)
                    vec_count = vec_df.drop_duplicates(keep='first').count().values
                    pairs_count = pairs_df.drop_duplicates(keep='first').count().values
                    if len(vec_count) == 0:
                       vec_number = 0
                    else:
                        vec_number = vec_count[0]
                    if len(pairs_count) == 0:
                       pair_number = 0
                    else:
                        pair_number = pairs_count[0]
                    print(pairs_df)
                    print(str(pair_number) + '###########')
                    print(vec_df)
                    print(str(vec_number) + '***************')
                    ff.write(DescriptorName + '|' + str(pair_number) + '|' + str(vec_number) + '|' + str(year) + '\n')

# calculate the super_concepts, sub_concepts, and siblings evolved in the last version.
def region_stability():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_negative_region_stability.txt', 'a') as ff:
        for year in range(2004, 2017):
            evolution = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_negative_samples_1.txt', sep='|',header=None)
            evolution.columns = ['DescriptorName', 'year', 'label', 'tre']
            evolution['id'] = evolution['DescriptorName'].map(str)+'|'+evolution['year'].map(str)
            # print(evolution['id'].drop_duplicates(keep='first').count())
            tree = pd.read_csv('/home/lab109/data/MeSH_2001_2020/tree/mesh_net' + str(year - 1) + '.txt', sep='|',
                               header=None)
            tree.columns = ['tree_start', 'tree_end', 'start', 'end', 'label']
            siblings = tree[tree['label'] == 'b-b'].loc[:, ['start', 'end']]
            # print(siblings[siblings['end'] == 'Microtubule-Organizing Center'])
            # print(siblings[siblings['start'] == 'Microtubule-Organizing Center'])
            children = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_children_name.txt', sep='|',header=None)
            children.columns = ['DescriptorName', 'children', 'year',  'class']
            children_ = children[children['year'] == year].loc[:, ['DescriptorName', 'children', 'year']]
            # print(children_[children_['DescriptorName'] == 'Microtubule-Organizing Center'])
            for id in evolution[evolution['year'] == year]['id'].values:
                DescriptorName = evolution[evolution['id'] == id]['DescriptorName'].values[0]
                trees_end = siblings[siblings['end'] == DescriptorName]['start']
                trees_end.columns = ['start']
                trees_start = siblings[siblings['start'] == DescriptorName]['end']
                trees_start.columns = ['start']
                sibling = pd.concat([trees_end, trees_start])
                # av include dv in children file, so drop_duplicates
                child = children_[children_['DescriptorName'] == DescriptorName]['children'].drop_duplicates(keep='first')
                child.columns = ['start']
                sibling = pd.concat([sibling, child], keys='start')
                # print(sibling)
                all_number = sibling.count()
                # print(all_number)
                evolution_pre = evolution[evolution['year'] == year-1]['DescriptorName'].values
                intersection = set(sibling.values).intersection(set(evolution_pre))
                inter_num = len(intersection)
                if all_number > 0:
                    print(id, all_number, inter_num, str(inter_num/all_number))
                    print(intersection)
                    ff.write(str(id)+'|'+str(inter_num/all_number)+'\n')
                else:
                    print(id, all_number, inter_num, '0')
                    ff.write(str(id) + '|' + '0' + '\n')


def region_stability_negative():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_negative_region_stability.txt', 'a') as ff:
        for year in range(2004, 2017):
            evolution = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_negative_samples_1.txt', sep='|',
                                    header=None)
            evolution.columns = ['DescriptorName', 'year', 'tre']
            evolution['id'] = evolution['DescriptorName'].map(str)+'|'+evolution['year'].map(str)
            # print(evolution['id'].drop_duplicates(keep='first').count())
            tree = pd.read_csv('/home/lab109/data/MeSH_2001_2020/tree/mesh_net' + str(year - 1) + '.txt', sep='|',
                               header=None)
            tree.columns = ['tree_start', 'tree_end', 'start', 'end', 'label']
            siblings = tree[tree['label'] == 'b-b'].loc[:, ['start', 'end']]
            children = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_children_name.txt', sep='|',
                               header=None)
            children.columns = ['DescriptorName', 'children', 'year',  'class']
            children_ = children[children['year'] == year].loc[:, ['DescriptorName', 'children', 'year']]
            # print(children_[children_['DescriptorName'] == 'Microtubule-Organizing Center'])
            for id in evolution[evolution['year'] == year]['id'].values:
                DescriptorName = evolution[evolution['id'] == id]['DescriptorName'].values[0]
                trees_end = siblings[siblings['end'] == DescriptorName]['start']
                trees_end.columns = ['start']
                trees_start = siblings[siblings['start'] == DescriptorName]['end']
                trees_start.columns = ['start']
                sibling = pd.concat([trees_end, trees_start])
                # av include dv in children file, so drop_duplicates
                child = children_[children_['DescriptorName'] == DescriptorName]['children'].drop_duplicates(keep='first')
                child.columns = ['start']
                sibling = pd.concat([sibling, child], keys='start')
                # print(sibling)
                all_number = sibling.count()
                # print(all_number)
                evolution_pre = evolution[evolution['year'] == year-1]['DescriptorName'].values
                intersection = set(sibling.values).intersection(set(evolution_pre))
                inter_num = len(intersection)
                if all_number > 0:
                    print(id, all_number, inter_num, str(inter_num/all_number))
                    print(intersection)
                    ff.write(str(id)+'|'+str(inter_num/all_number)+'\n')
                else:
                    print(id, all_number, inter_num, '0')
                    ff.write(str(id) + '|' + '0' + '\n')


def boost_rate(input):
    input = input.apply(lambda x: float(x))
    x_lis = []
    for x in range(1, input.count()+1):
        x_lis.append(x)
    X = np.array(x_lis)
    Y = np.array(input.values)
    def func(p, x):
        k, b = p
        return k * x + b
    def error(p, x, y):
        return func(p, x) - y
    p0 = [1, 20]
    Para = leastsq(error, p0, args=(X, Y))
    k, b = Para[0]
    return k


def mesh_ego_structure_feature_analysis():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_ego_structure_analysis_negative_rate.txt', 'a') as f:
        df = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_ego_structure_analysis_negative.txt', sep='|', header=None)
        df.columns = ['mesh_name', 'year', 'type', 'density', 'H2', 'H3', 'ego_number', 'yr']
        name_year = df.loc[:, ['mesh_name', 'year']].drop_duplicates(keep='first')
        print(name_year)
        for mesh_name, year in name_year.values:
            temp = df[df['mesh_name'] == mesh_name]
            mesh = temp[temp['year'] == year]
            type = mesh['type'].values[0]
            den = mesh[mesh['density'] != 'NAN']['density']
            H2 = mesh[mesh['H2'] != 'NAN']['H2']
            H3 = mesh[mesh['H3'] != 'NAN']['H3']
            ego_num = mesh[mesh['ego_number'] != 'NAN']['ego_number']

            if den.count() < 2:
                den_value = 'NAN'
            else:
                den_value = boost_rate(den)

            if H2.count() < 2:
                H2_value = 'NAN'
            else:
                H2_value = boost_rate(H2)

            if H3.count() < 2:
                H3_value = 'NAN'
            else:
                H3_value = boost_rate(H3)

            if ego_num.count() < 2:
                ego_num_value = 'NAN'
            else:
                ego_num_value = boost_rate(ego_num)
            print(mesh_name, year, type, den_value, H2_value, H3_value, ego_num_value)
            f.write(str(mesh_name)+'|'+str(year)+'|'+str(type)+'|'+str(den_value)+'|'+str(H2_value)+'|'+str(H3_value)+'|'+str(ego_num_value)+'\n')

def mesh_ego_structure_feature_analysis_n():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_ego_structure_analysis_negative_rate.txt', 'a') as f:
        df = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_ego_structure_analysis_negative.txt', sep='|', header=None)
        df.columns = ['mesh_name', 'year', 'density', 'H2', 'H3', 'ego_number', 'yr']
        name_year = df.loc[:, ['mesh_name', 'year']].drop_duplicates(keep='first')
        print(name_year)
        for mesh_name, year in name_year.values:
            temp = df[df['mesh_name'] == mesh_name]
            mesh = temp[temp['year'] == year]
            # type = mesh['type'].values[0]
            den = mesh[mesh['density'] != 'NAN']['density']
            H2 = mesh[mesh['H2'] != 'NAN']['H2']
            H3 = mesh[mesh['H3'] != 'NAN']['H3']
            ego_num = mesh[mesh['ego_number'] != 'NAN']['ego_number']

            if den.count() < 2:
                den_value = 'NAN'
            else:
                den_value = boost_rate(den)

            if H2.count() < 2:
                H2_value = 'NAN'
            else:
                H2_value = boost_rate(H2)

            if H3.count() < 2:
                H3_value = 'NAN'
            else:
                H3_value = boost_rate(H3)

            if ego_num.count() < 2:
                ego_num_value = 'NAN'
            else:
                ego_num_value = boost_rate(ego_num)
            print(mesh_name, year, den_value, H2_value, H3_value, ego_num_value)
            f.write(str(mesh_name)+'|'+str(year)+'|'+str(den_value)+'|'+str(H2_value)+'|'+str(H3_value)+'|'+str(ego_num_value)+'\n')

# construct knowledge graph
def knowledge_graph(year):
    sql = '''
            SELECT t6.DescriptorName_start,
           t6.DescriptorName_end,
           t6.weight
    FROM (
             SELECT t5.DescriptorName_start,
                    t5.DescriptorName_end,
                    ((toInt32(t5.weight) / toInt32(t5.start_count)) + (toInt32(t5.weight) / toInt32(t5.end_count))) /
                    2 as weight
             FROM (SELECT t3.DescriptorName_start, t3.DescriptorName_end, t3.weight, t3.start_count, t4.count as end_count
                   from (
                            SELECT t1.DescriptorName_start,
                                   t1.DescriptorName_end,
                                   t2.DescriptorName,
                                   t2.count as start_count,
                                   t1.weight
                            FROM (SELECT t1.DescriptorName_start, t1.DescriptorName_end, count() as weight
                                  FROM A15_keyword_pairs t1
                                  where toInt32(t1.Year) <= ''' + str(year) + '''
                                  group by t1.DescriptorName_start, t1.DescriptorName_end) t1
                                     inner join V02_DescriptorName_count t2
                                                on t1.DescriptorName_start = t2.DescriptorName ) t3
                            inner join V02_DescriptorName_count t4
                                       on t3.DescriptorName_end = t4.DescriptorName) t5
             ) t6 where t6.weight > 0.00015 order by t6.weight ASC;
                  '''
    data = click_server(sql)
    pairs = [tuple(x) for x in data.values]
    g = ig.Graph.TupleList(pairs, edge_attrs="weight", directed=False)
    print(ig.summary(g))
    return g

# construct knowledge network
def nx_network(year):
    sql = '''
        SELECT t6.DescriptorName_start,
       t6.DescriptorName_end,
       t6.weight
FROM (
         SELECT t5.DescriptorName_start,
                t5.DescriptorName_end,
                ((toInt32(t5.weight) / toInt32(t5.start_count)) + (toInt32(t5.weight) / toInt32(t5.end_count))) /
                2 as weight
         FROM (SELECT t3.DescriptorName_start, t3.DescriptorName_end, t3.weight, t3.start_count, t4.count as end_count
               from (
                        SELECT t1.DescriptorName_start,
                               t1.DescriptorName_end,
                               t2.DescriptorName,
                               t2.count as start_count,
                               t1.weight
                        FROM (SELECT t1.DescriptorName_start, t1.DescriptorName_end, count() as weight
                              FROM A15_keyword_pairs t1
                              where toInt32(t1.Year) = ''' + str(year) + '''
                              group by t1.DescriptorName_start, t1.DescriptorName_end) t1
                                 inner join V02_DescriptorName_count t2
                                            on t1.DescriptorName_start = t2.DescriptorName ) t3
                        inner join V02_DescriptorName_count t4
                                   on t3.DescriptorName_end = t4.DescriptorName) t5
         ) t6 where t6.weight > 0.00015 order by t6.weight ASC;
              '''
    data = click_server(sql)
    edges = data
    g = nx.Graph()
    g.add_weighted_edges_from(tuple(edges.values), directed=False)
    print(g.size())
    return g

def evolution_mesh_ego_structure_analysis(year, i):
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_ego_structure_analysis_negative.txt', 'a') as ff:
        sql = '''
                SELECT t1.DescriptorName, year FROM V02_1_non_evolution t1 where toInt32(t1.year) >=2004
              '''
        mesh = click_server(sql)
        print(mesh)
        mesh_names = mesh[mesh['year'] == str(year)]['DescriptorName']
        print(mesh_names)
        for yr in range(year-5, year):
            g = nx_network(yr)
            all_nodes = g.nodes()
            for mesh_name in mesh_names.values:
                # print(all_nodes)
                if mesh_name in all_nodes:
                    print(mesh_name)
                    ego = nx.ego_graph(g, str(mesh_name))
                    ego_number = len(ego.nodes())
                    gg = ig.Graph.from_networkx(ego)
                    density = gg.density()
                    H2 = gg.triad_census().t201
                    H3 = gg.triad_census().t300
                    print(density, H2, H3)
                    print(mesh_name, year, density, H2, H3, ego_number, yr)
                    print('Thread.....' + str(i))
                    ff.write(str(mesh_name) + '|' + str(year) + '|' + str(density)+ '|' + str(H2) + '|' + str(H3) + '|' + str(ego_number) +'|'+ str(yr)+ '\n')
                else:
                    print(mesh_name, year,'NAN', 'NAN', 'NAN', 'NAN', yr)
                    print('Thread.....' + str(i))
                    ff.write(str(mesh_name)+'|'+str(year)+'|'+'NAN'+'|'+'NAN'+'|' + 'NAN' + '|' + 'NAN' + '|' + str(yr) + '\n')

def evolution_mesh_structure_analysis():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_structure_analysis_weight_negative.txt', 'a') as f:
        sql = '''
            SELECT t1.DescriptorName, year FROM V02_1_non_evolution t1 where toInt32(t1.year) >=2004
              '''
        mesh = click_server(sql)
        print(mesh)
        for year in range(2004, 2017):
            g = knowledge_graph(year-1)
            mesh_names = mesh[mesh['year'] == str(year)]['DescriptorName']
            print(mesh_names)
            all_nodes = g.vs['name']
            all_number = len(all_nodes)
            print(all_number)
            print(all_nodes)
            for mesh_name in mesh_names.values:
                print(mesh_name)
                if mesh_name in all_nodes:
                    print(mesh_name)
                    degree = g.vs.find(mesh_name).degree()
                    # print(avg_degree)
                    closeness = g.vs.find(mesh_name).closeness()
                    clustering = g.transitivity_local_undirected(mesh_name, weights=g.es["weight"])
                    pagerank = g.vs.find(mesh_name).pagerank()
                    print(mesh_name, year, degree, closeness, clustering, pagerank)
                    f.write(str(mesh_name) + '|' + str(year) + '|' + str(degree)+ '|' + str(closeness) + '|' + str(clustering) +'|'+ str(pagerank)+ '\n')
                else:
                    print(mesh_name, year,'NAN', 'NAN', 'NAN', 'NAN')
                    f.write(str(mesh_name)+'|'+str(year)+'|'+'NAN'+'|'+'NAN'+'|' + 'NAN' + '|' + 'NAN' + '\n')

def multi_thread():
    start = 2003
    for i in range(1, 14):
        t = threading.Thread(target=evolution_mesh_ego_structure_analysis, args=(start+i, i))
        t.start()

if __name__ == '__main__':
    mesh_ego_structure_feature_analysis_n()