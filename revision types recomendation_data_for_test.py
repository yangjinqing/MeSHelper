# -*- coding: UTF-8 -*-
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from clickhouse_driver import Client as click_client
import re
import multi_classify_descriptors_ROC as cl_roc

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


def NAN_to_num(x, num):
    if x =='NAN':
        r = float(num)
    else:
        r = x
    return float(r)

def data_agg_input():
    # publication #
    Sql_mesh ='SELECT t. DescriptorName,t.year as year ,t.type as type  FROM V02_evolution_types t where toInt32(t.year) >=2004'
    SQL_pub = 'SELECT t. DescriptorName,t.year as year,t.count FROM V02_Pub_number_evolution t'
    SQL_pub_dv = 'SELECT t.DesFather as DescriptorName,t.year as year,t.count  FROM V02_Pub_number_children t'
    SQL_pub_av = 'SELECT t.DesFather as DescriptorName,t.year as year,t.count FROM V02_Pub_number_descendants t'
    mesh = click_server(Sql_mesh)
    Pub = click_server(SQL_pub)
    Pub_dv = click_server(SQL_pub_dv)
    Pub_av = click_server(SQL_pub_av)
    Publication = mesh.merge(Pub, on=['DescriptorName', 'year'], how='left').merge(Pub_dv, on=['DescriptorName', 'year'], how='left').merge(Pub_av, on=['DescriptorName', 'year'], how='left')
    Publication.columns = ['DescriptorName', 'year', 'type', 'pub_count', 'pub_count_av', 'pub_count_dv']
    Publication = Publication.fillna(value=0)
    # Publication.to_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/publication.csv')

    #citation#
    SQL_citation= 'SELECT t. DescriptorName,t.year as year,t.count  FROM V02_citation_count_evolution t'
    SQL_citation_dv = 'SELECT t.DesFather as DescriptorName,t.year as year,t.count FROM V02_citation_count_children t'
    SQL_citation_av = 'SELECT t.DesFather as DescriptorName,t.year as year,t.count FROM V02_citation_count_descendants t'
    citation = click_server(SQL_citation)
    citation_dv = click_server(SQL_citation_dv)
    citation_av = click_server(SQL_citation_av)
    citation = mesh.merge(citation, on=['DescriptorName', 'year'], how='left').merge(citation_av, on=['DescriptorName', 'year'], how='left').merge(  citation_dv, on=['DescriptorName', 'year'], how='left')
    citation.columns = ['DescriptorName', 'year', 'type', 'citation_pub', 'citation_av', 'citation_dv']
    citation = citation.loc[:, ['DescriptorName', 'year', 'citation_pub', 'citation_av', 'citation_dv']]
    citation = citation.fillna(value=0)

    # structural feature #
    sibling = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_siblings_positive.txt', sep='|', header=None)
    # print(sibling)
    sibling.columns = ['DescriptorName', 'count', 'year']
    children = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_evolution_children_positive.txt', sep='|', header=None)
    children.columns = ['DescriptorName', 'count_dv', 'count_av', 'year']
    stability = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/mesh_positive_region_stability.txt', sep='|', header=None)
    stability.columns = ['DescriptorName', 'year', 'count_sta']
    sibling['year'] = sibling['year'].apply(str)
    children['year'] = children['year'].apply(str)
    stability['year'] = stability['year'].apply(str)
    structure = mesh.merge(sibling, how='left', on=['DescriptorName', 'year']).merge(children, how='left', on=['DescriptorName', 'year']).merge(stability, how='left', on=['DescriptorName', 'year'])
    structure.columns = ['DescriptorName', 'year', 'type', 'count_sib', 'count_dv', 'count_av', 'count_sta']
    structure = structure.loc[:, ['DescriptorName', 'year', 'count_sib', 'count_dv', 'count_av', 'count_sta']]
    structure = structure.fillna(value=0)
    print(structure.count())

    # dynamic_semantic_network
    dsn = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_structure_analysis_weight.txt', sep='|', header=None)
    dsn.columns = ['DescriptorName', 'year', 'type', 'degree', 'closeness', 'clustering', 'pagerank']
    dsn = dsn.loc[:, ['DescriptorName', 'year', 'degree', 'closeness', 'clustering', 'pagerank']]
    dsn['year'] = dsn['year'].apply(str)
    # dsn['type'] = dsn['type'].apply(str)

    dsn['degree'] = dsn['degree'].apply(lambda x: NAN_to_num(x, 0))
    dsn['closeness'] = dsn['closeness'].apply(lambda x: NAN_to_num(x, 0))
    dsn['clustering'] = dsn['clustering'].apply(lambda x: NAN_to_num(x, 0))
    dsn['pagerank'] = dsn['pagerank'].apply(lambda x: NAN_to_num(x, 0))
    print(dsn)

    # temporal rate#
    tr = pd.read_csv('/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_ego_structure_analysis_rate2.txt', sep='|', header=None)
    tr.columns = ['DescriptorName', 'year', 'type', 'density_rate', 'H2_rate', 'H3_rate', 'ego_number_rate']
    tr['year'] = tr['year'].apply(str)
    tr = tr.loc[:, ['DescriptorName', 'year', 'density_rate', 'H2_rate', 'H3_rate', 'ego_number_rate']]
    tr['density_rate'] = tr['density_rate'].apply(lambda x: NAN_to_num(x, 0))
    tr['H2_rate'] = tr['H2_rate'].apply(lambda x: NAN_to_num(x, 0))
    tr['H3_rate'] = tr['H3_rate'].apply(lambda x: NAN_to_num(x, 0))
    tr['ego_number_rate'] = tr['ego_number_rate'].apply(lambda x: NAN_to_num(x, 0))
    # tr = tr.merge(evolution_age, on=['DescriptorName', 'year'], how='left')
    print(tr)

    data_positive = Publication.merge(citation, how='left', on=['DescriptorName', 'year']).merge(structure, how='left', on=['DescriptorName', 'year']).merge(tr, how='left', on=['DescriptorName', 'year']).merge(dsn, how='left', on=['DescriptorName', 'year'])
    # print(data_positive.groupby('type').count())
    print('--------E----------')
    data_positive_E = data_positive[data_positive['type'] == 'E']
    print(data_positive_E.count())
    data_positive_E['re'] = 0
    print('---------C---------')
    data_positive_C = data_positive[data_positive['type'] == 'C']
    print(data_positive_C.count())
    data_positive_C['re'] = 1
    print('---------M---------')
    data_positive_M = data_positive[data_positive['type'] == 'M']
    print(data_positive_M.count())
    print('--------R----------')
    data_positive_M['re'] = 2
    data_positive_R = data_positive[data_positive['type'] == 'R']
    print(data_positive_R)
    data_positive_R['re'] = 2
    data = pd.concat([data_positive_E, data_positive_C, data_positive_M, data_positive_R])
    print(data)
    data = data.fillna(value=0)
    # print(data)
    print(data.groupby('type').count())
    data['year'] = data['year'].apply(int)
    # print(data.columns)
    data_test=data[data['year'] <= 2016]
    train = data_test[data_test['year'] <= 2014].loc[:, ['pub_count', 'pub_count_av', 'pub_count_dv','citation_pub', 'citation_av', 'citation_dv', 'count_sib', 'count_dv', 'count_av', 'count_sta','degree', 'closeness', 'clustering', 'pagerank', 'density_rate', 'H2_rate', 'H3_rate', 'ego_number_rate', 're']]
    test = data_test[data_test['year'] >= 2015].loc[:, ['type','pub_count', 'pub_count_av', 'pub_count_dv', 'citation_pub', 'citation_av', 'citation_dv', 'count_sib', 'count_dv', 'count_av', 'count_sta', 'degree', 'closeness', 'clustering', 'pagerank', 'density_rate', 'H2_rate', 'H3_rate', 'ego_number_rate', 're']]
    feature = ['pub_count', 'pub_count_av', 'pub_count_dv', 'citation_pub', 'citation_av', 'citation_dv', 'count_sib', 'count_dv', 'count_av', 'count_sta', 'degree', 'closeness', 'clustering', 'pagerank', 'density_rate','H2_rate', 'H3_rate', 'ego_number_rate']
    smote_enn = SMOTEENN(random_state=0)
    smote = SMOTE(random_state=0)
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = smote.fit_resample(train.loc[:, feature], train['re'])
    cl_roc.multi_classification(X_resampled, y_resampled, test, feature)
    # ee.multi_classification(train.loc[:, feature], train['re'], test, feature)
    # for para1 in range(1, 100):
    #     for para2 in ['l2']:
        # for para2 in ['uniform','distance']:
        #     pee.multi_classification(train.loc[:, feature], train['re'], test, feature,para1,para2, 'Linear_SVC')
        #     clp.multi_classification(X_resampled, y_resampled, test, feature,para1, para2, 'Linear_SVC')

    # Me.explain_multi_RFs(X_resampled, y_resampled)
    # print(X_resampled, y_resampled)

if __name__ == '__main__':
    data_agg_input()