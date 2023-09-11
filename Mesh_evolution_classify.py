# -*- coding: UTF-8 -*-
import pymysql
# import igraph as g
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from clickhouse_driver import Client as click_client
import re
import classify_descriptors_revise as cl
import classify_descriptors_roc as rc
import Model_explainability as Me

def NAN_to_num(x, num):
    if x == 'NAN':
        r = num
    else:
        r = x
    return r


# access clickhouse database
def click_server(sql):
    chs_host = 'localhost'
    chs_user = 'default'
    chs_pwd = 'root'
    chs_port = '9001'
    chs_database = 'pubmed20'
    client = click_client(host=chs_host, port=chs_port, user=chs_user, password=chs_pwd, database=chs_database,
                          send_receive_timeout=5)
    # ans = client.execute(query=query, with_column_types=True)
    data, columns = client.execute(
        sql, columnar=True, with_column_types=True)
    df = pd.DataFrame({re.sub(r'\W', '_', col[0]): d for d, col in zip(data, columns)})
    return df


def test_data():
    ################
    # positive######
    ################
    # publication #
    Sql_mesh = 'SELECT t. DescriptorName,t.year as year FROM V02_evolution_types t where toInt32(t.year) >=2004'
    SQL_pub = 'SELECT t. DescriptorName,t.year as year,t.count FROM V02_Pub_number_evolution t'
    SQL_pub_dv = 'SELECT t.DesFather as DescriptorName,t.year as year,t.count  FROM V02_Pub_number_children t'
    SQL_pub_av = 'SELECT t.DesFather as DescriptorName,t.year as year,t.count FROM V02_Pub_number_descendants t'
    mesh = click_server(Sql_mesh)
    Pub = click_server(SQL_pub)
    Pub_dv = click_server(SQL_pub_dv)
    Pub_av = click_server(SQL_pub_av)
    Publication = mesh.merge(Pub, on=['DescriptorName', 'year'], how='left').merge(Pub_dv, on=['DescriptorName', 'year'], how='left').merge(Pub_av, on=['DescriptorName', 'year'], how='left')
    Publication.columns = ['DescriptorName', 'year', 'pub_count', 'pub_count_av', 'pub_count_dv']
    Publication=Publication.fillna(0)
    Publication['year'] = Publication['year'].apply(str)
    # print(Publication)

    # # citation #
    SQL_citation = 'SELECT t. DescriptorName,t.year as year,t.count  FROM V02_citation_count_evolution t'
    SQL_citation_dv = 'SELECT t.DesFather as DescriptorName,t.year as year,t.count FROM V02_citation_count_children t'
    SQL_citation_av = 'SELECT t.DesFather as DescriptorName,t.year as year,t.count FROM V02_citation_count_descendants t'
    citation = click_server(SQL_citation)
    citation_dv = click_server(SQL_citation_dv)
    citation_av = click_server(SQL_citation_av)
    citation = mesh.merge(citation, on=['DescriptorName', 'year'], how='left').merge(citation_av, on=['DescriptorName', 'year'],how='left').merge(citation_dv, on=['DescriptorName', 'year'], how='left')
    citation.columns = ['DescriptorName', 'year', 'citation_pub', 'citation_av', 'citation_dv']
    citation=citation.fillna(0)
    # print(citation)

    # positive structural feature #
    sibling = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_siblings_positive.txt',  sep='|', header=None)
    # print(sibling)
    sibling.columns = ['DescriptorName', 'count', 'year']
    children = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_children_positive.txt',
                           sep='|', header=None)
    children.columns = ['DescriptorName', 'count_dv', 'count_av', 'year']
    stability = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_positive_region_stability.txt',
                            sep='|', header=None)
    stability.columns = ['DescriptorName', 'year', 'count_sta']
    sibling['year'] = sibling['year'].apply(str)
    children['year'] = children['year'].apply(str)
    stability['year'] = stability['year'].apply(str)
    structure = mesh.merge(sibling, how='left', on=['DescriptorName', 'year']).merge(children, how='left',  on=['DescriptorName', 'year']).merge(stability, how='left', on=['DescriptorName', 'year'])
    structure.columns = ['DescriptorName', 'year', 'count_sib', 'count_dv', 'count_av', 'count_sta']
    structure=structure.fillna(0)
    # print(structure)

    # dynamic_semantic_network
    dsn = pd.read_csv(
        '/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_structure_analysis_weight.txt', sep='|', header=None)
    dsn.columns = ['DescriptorName', 'year', 'type', 'degree', 'closeness', 'clustering', 'pagerank']
    dsn = dsn.loc[:, ['DescriptorName', 'year', 'degree', 'closeness', 'clustering', 'pagerank']]
    dsn['year'] = dsn['year'].apply(str)
    # dsn['type'] = dsn['type'].apply(str)

    dsn['degree'] = dsn['degree'].apply(lambda x: NAN_to_num(x, 0))
    dsn['degree'] = dsn['degree'].apply(float)

    dsn['closeness'] = dsn['closeness'].apply(lambda x: NAN_to_num(x, 0))
    dsn['closeness'] = dsn['closeness'].apply(float)

    dsn['clustering'] = dsn['clustering'].apply(lambda x: NAN_to_num(x, 0))
    dsn['clustering'] = dsn['clustering'].apply(float)

    dsn['pagerank'] = dsn['pagerank'].apply(lambda x: NAN_to_num(x, 0))
    dsn['pagerank'] = dsn['pagerank'].apply(float)

    print(dsn)
    #
    # temporal rate#
    tr = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_ego_structure_analysis_rate2.txt', sep='|', header=None)
    tr.columns = ['DescriptorName', 'year', 'type', 'density_rate', 'H2_rate', 'H3_rate', 'ego_number_rate']
    tr = tr.loc[:, ['DescriptorName', 'year', 'density_rate', 'H2_rate', 'H3_rate', 'ego_number_rate']]
    tr['year'] = tr['year'].apply(str)
    tr = tr.loc[:, ['DescriptorName', 'year', 'density_rate', 'H2_rate', 'H3_rate', 'ego_number_rate']]
    tr['density_rate'] = tr['density_rate'].apply(lambda x: NAN_to_num(x, 0))
    tr['density_rate'] = tr['density_rate'].apply(float)

    tr['H2_rate'] = tr['H2_rate'].apply(lambda x: NAN_to_num(x, 0))
    tr['H2_rate'] = tr['H2_rate'].apply(float)

    tr['H3_rate'] = tr['H3_rate'].apply(lambda x: NAN_to_num(x, 0))
    tr['H3_rate'] = tr['H3_rate'].apply(float)

    tr['ego_number_rate'] = tr['ego_number_rate'].apply(lambda x: NAN_to_num(x, 0))
    tr['ego_number_rate'] = tr['ego_number_rate'].apply(float)

    # tr = tr.merge(evolution_age, on=['DescriptorName', 'year'], how='left')
    # print(tr)
    #
    # # ################
    # # # negative######
    # # ################

    Sql_mesh2 = 'SELECT t. DescriptorName,t.year as year FROM V02_1_non_evolution t'
    SQL_pub2 = 'SELECT t. DescriptorName,t.year as year,t.count FROM V02_1_Pub_number_non_evolution t'
    SQL_pub_dv2 = 'SELECT t.DesFather as DescriptorName,t.year as year,t.count  FROM V02_1_Pub_number_children t'
    SQL_pub_av2 = 'SELECT t.DesFather as DescriptorName,t.year as year,t.count FROM V02_1_Pub_number_descendants t'
    mesh2 = click_server(Sql_mesh2)
    Pub2 = click_server(SQL_pub2)
    Pub_dv2 = click_server(SQL_pub_dv2)
    Pub_av2 = click_server(SQL_pub_av2)
    Publication2 = mesh2.merge(Pub2, on=['DescriptorName', 'year'], how='left').merge(Pub_dv2, on=['DescriptorName', 'year'], how='left').merge(Pub_av2, on=['DescriptorName', 'year'], how='left')
    Publication2.columns = ['DescriptorName', 'year', 'pub_count', 'pub_count_av', 'pub_count_dv']
    Publication2['year'] = Publication2['year'].apply(str)
    # print(Publication2)

    # # citation #
    SQL_citation2 = 'SELECT t. DescriptorName,t.year as year,t.count  FROM V02_1_citation_count_non_evolution t'
    SQL_citation_dv2 = 'SELECT t.DesFather as DescriptorName,t.year as year,t.count FROM V02_1_citation_count_children t'
    SQL_citation_av2 = 'SELECT t.DesFather as DescriptorName,t.year as year,t.count FROM V02_1_citation_count_descendants t'
    citation2 = click_server(SQL_citation2)
    citation_dv2 = click_server(SQL_citation_dv2)
    citation_av2 = click_server(SQL_citation_av2)
    citation2 = mesh2.merge(citation2, on=['DescriptorName', 'year'], how='left').merge(citation_av2, on=['DescriptorName', 'year'],how='left').merge(citation_dv2, on=['DescriptorName', 'year'], how='left')
    citation2.columns = ['DescriptorName', 'year', 'citation_pub', 'citation_av', 'citation_dv']
    # print(citation2)

    # positive structural feature #
    sibling2 = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_siblings_negative_1.txt', sep='|', header=None)
    # print(sibling)
    sibling2.columns = ['DescriptorName', 'count', 'year']
    children2 = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_children_negative_1.txt',
                            sep='|', header=None)
    children2.columns = ['DescriptorName', 'count_dv', 'count_av', 'year']
    stability2 = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_negative_region_stability.txt',
                             sep='|', header=None)
    stability2.columns = ['DescriptorName', 'year', 'count_sta']
    sibling2['year'] = sibling2['year'].apply(str)
    children2['year'] = children2['year'].apply(str)
    stability2['year'] = stability2['year'].apply(str)
    structure2 = mesh2.merge(sibling2, how='left', on=['DescriptorName', 'year']).merge(children2, how='left', on=['DescriptorName','year']).merge(stability2, how='left', on=['DescriptorName', 'year'])
    structure2.columns = ['DescriptorName', 'year', 'count_sib', 'count_dv', 'count_av', 'count_sta']
    # print(structure2)

    # dynamic_semantic_network_negative
    dsn2 = pd.read_csv( '/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_structure_analysis_weight_negative.txt', sep='|',
        header=None)
    dsn2.columns = ['DescriptorName', 'year', 'degree', 'closeness', 'clustering', 'pagerank']
    dsn2['year'] = dsn2['year'].apply(str)
    dsn2['degree'] = dsn2['degree'].apply(lambda x: NAN_to_num(x, 0))
    dsn2['degree'] = dsn2['degree'].apply(float)

    dsn2['closeness'] = dsn2['closeness'].apply(lambda x: NAN_to_num(x, 0))
    dsn2['closeness'] = dsn2['closeness'].apply(float)

    dsn2['clustering'] = dsn2['clustering'].apply(lambda x: NAN_to_num(x, 0))
    dsn2['clustering'] = dsn2['clustering'].apply(float)

    dsn2['pagerank'] = dsn2['pagerank'].apply(lambda x: NAN_to_num(x, 0))
    dsn2['pagerank'] = dsn2['pagerank'].apply(float)
    # print(dsn2)

    # temporal rate#
    tr2 = pd.read_csv(
        '/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_ego_structure_analysis_negative_rate.txt',sep='|', header=None)
    tr2.columns = ['DescriptorName', 'year', 'density_rate', 'H2_rate', 'H3_rate', 'ego_number_rate']
    tr2['year'] = tr2['year'].apply(str)
    tr2 = tr2.loc[:, ['DescriptorName', 'year', 'density_rate', 'H2_rate', 'H3_rate', 'ego_number_rate']]
    tr2['density_rate'] = tr2['density_rate'].apply(lambda x: NAN_to_num(x, 0))
    tr2['density_rate']=tr2['density_rate'].apply(float)

    tr2['H2_rate'] = tr2['H2_rate'].apply(lambda x: NAN_to_num(x, 0))
    tr2['H2_rate'] = tr2['H2_rate'].apply(float)

    tr2['H3_rate'] = tr2['H3_rate'].apply(lambda x: NAN_to_num(x, 0))
    tr2['H3_rate'] = tr2['H3_rate'].apply(float)

    tr2['ego_number_rate'] = tr2['ego_number_rate'].apply(lambda x: NAN_to_num(x, 0))
    tr2['ego_number_rate'] = tr2['ego_number_rate'].apply(float)
    # tr = tr.merge(evolution_age, on=['DescriptorName', 'year'], how='left')
    # print(tr)

    data_positive = Publication.merge(citation, on=['DescriptorName', 'year']).merge(structure, on=['DescriptorName', 'year']).merge(dsn,on=['DescriptorName','year']).merge(tr,on=['DescriptorName','year'])
    data_negative = Publication2.merge(citation2, on=['DescriptorName', 'year']).merge(structure2, on=['DescriptorName', 'year']).merge( dsn2, on=['DescriptorName', 'year']).merge(tr2,on=['DescriptorName','year'])
    data_positive['re'] = 1
    data_negative['re'] = 0

    data_positive = data_positive.fillna(0)
    data_negative = data_negative.fillna(0)
    print(data_positive.loc[:, ['degree', 'closeness', 'clustering', 'pagerank', 'density_rate', 'H2_rate', 'H3_rate', 'ego_number_rate']])
    print('***************************************')
    print(data_negative.loc[:, ['degree', 'closeness', 'clustering', 'pagerank', 'density_rate', 'H2_rate', 'H3_rate', 'ego_number_rate']])
    print('****************************************')
    data = pd.concat([data_positive, data_negative])
    data['year'] = data['year'].apply(int)
    # print(data.columns)

    train = data[data['year'] <= 2014].loc[:, ['pub_count', 'pub_count_av', 'pub_count_dv',
                                                 'citation_pub', 'citation_av', 'citation_dv', 'count_sib', 'count_dv',
                                                 'count_av', 'count_sta', 'degree', 'closeness', 'clustering',
                                                 'pagerank', 'density_rate', 'H2_rate', 'H3_rate', 'ego_number_rate', 're']]
    test = data[data['year'] >= 2015].loc[:,
           ['pub_count', 'pub_count_av', 'pub_count_dv', 'citation_pub', 'citation_av', 'citation_dv', 'count_sib',
            'count_dv', 'count_av', 'count_sta', 'degree', 'closeness', 'clustering', 'pagerank', 'density_rate', 'H2_rate', 'H3_rate', 'ego_number_rate', 're']]
    # print(train.columns)

    feature = ['pub_count', 'pub_count_av', 'pub_count_dv', 'citation_pub', 'citation_av', 'citation_dv', 'count_sib', 'count_dv','count_av', 'count_sta', 'degree', 'closeness', 'clustering', 'pagerank', 'density_rate', 'H2_rate', 'H3_rate', 'ego_number_rate']

    rc.classify_descriptor(train, test, feature)
    # Me.explain_RFs(train, feature)
    # Me.explain_RFs_for_single(train, feature)
if __name__ == '__main__':
    test_data()
