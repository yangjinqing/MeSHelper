# -*- coding: UTF-8 -*-
import pandas as pd
from itertools import combinations
import igraph as ig
import networkx as nx
from clickhouse_driver import Client as click_client
import re
import threading

def click_server(sql):
    chs_host = 'localhost'
    chs_user = 'default'
    chs_pwd = 'root'
    chs_port = '9001'
    chs_database = 'pubmed20'
    client = click_client(host=chs_host, port=chs_port, user=chs_user, password=chs_pwd, database=chs_database,
                          send_receive_timeout=5)
    # ans = client.execute(query=sql, with_column_types=True)
    data, columns = client.execute(sql, columnar=True, with_column_types=True)
    df = pd.DataFrame({re.sub(r'\W', '_', col[0]): d for d, col in zip(data, columns)})
    return df

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
    # return g

# construct knowledge network
def nx_network():
    # year = 2019
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
                              where toInt32(t1.Year) <= 2019
                              group by t1.DescriptorName_start, t1.DescriptorName_end) t1
                                 inner join V02_DescriptorName_count t2
                                            on t1.DescriptorName_start = t2.DescriptorName ) t3
                        inner join V02_DescriptorName_count t4
                                   on t3.DescriptorName_end = t4.DescriptorName) t5
         ) t6;
              '''
    data = click_server(sql)
    # print(refer_pairs)
    edges = data
    # print(tuple(edges.values))
    g = nx.Graph()
    g.add_weighted_edges_from(tuple(edges.values), directed=False)
    print(g.size())
    # return g


def keyword_occurrence():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/Mesh_occurrence_1965_1970.txt',
              'a') as ff:
        # stop in 1966
        sql = '''SELECT t1.PMID, DescriptorName, t2.publicationyear
FROM A06_MeshHeadingList t1
         inner join A01_Articles_publicationyear t2 on t1.PMID = t2.PMID
where toInt32(t2.publicationyear) between 1985 and 1993
order by t2.publicationyear ASC;'''
        data = click_server(sql)
        for PMID in data['PMID'].drop_duplicates(keep='first'):
            comb = data[data['PMID'] == PMID]['DescriptorName']
            comb = comb.sort_values(0, ascending=True)
            mesh_comb = combinations(comb, 2)
            year = data[data['PMID'] == PMID]['publicationyear'].drop_duplicates(keep='first').values[0]
            for keyword_c in mesh_comb:
                print(str(PMID) + '|' + str(keyword_c[0]).strip() + '|' + str(keyword_c[1]).strip() + '|' + str(year))
                ff.write(str(PMID) + '|' + str(keyword_c[0]).strip() + '|' + str(keyword_c[1]).strip() + '|' + str(
                    year) + '\n')


def evolution_dynamic_structure_analysis():
    with open(
            '/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_dynamic_structure_analysis_2016.txt',
            'a') as ff:
        sql = '''
            SELECT t1.DescriptorName, year, type FROM V02_evolution_types t1 where toInt32(t1.year) >=2004
              '''
        mesh = click_server(sql)
        print(mesh)
        for yr in range(2016, 2017):
            mesh_names = mesh[mesh['year'] == str(yr)].loc[:, ['DescriptorName', 'type']]
            print(mesh_names)
            for year in range(yr - 4, yr):
                g = knowledge_graph(year)
                all_nodes = g.vs['name']
                for mesh_name, type in mesh_names.values:
                    if mesh_name in all_nodes:
                        degree = g.vs.find(mesh_name).degree()
                        print(degree)
                        closeness = g.vs.find(mesh_name).closeness()
                        print(closeness)
                        clustering = g.transitivity_local_undirected(mesh_name, weights=g.es["weight"])
                        print(clustering)
                        pagerank = g.vs.find(mesh_name).pagerank()
                        print(pagerank)
                        strength = g.strength(mesh_name, weights=g.es["weight"])
                        print(strength)
                        print(mesh_name, yr, type, year, degree, closeness, clustering, pagerank, strength)
                        ff.write(str(mesh_name) + '|' + str(yr) + '|' + str(type) + '|' + str(year) + '|' + str(
                            degree) + '|' + str(closeness) + '|' + str(clustering) + '|' + str(pagerank) + '|' + str(
                            strength) + '\n')
                    else:
                        print(mesh_name, yr, type, year, 'NAN', 'NAN', 'NAN', 'NAN', 'NAN')
                        ff.write(str(mesh_name) + '|' + str(yr) + '|' + str(type) + '|' + str(year) + '|' + str(
                            'NAN') + '|' + str('NAN') + '|' + str('NAN') + '|' + str('NAN') + '|' + str('NAN') + '\n')


def DescriptorName_UI_occurrence():
    with open(
            '/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/DescriptorName_UI_occurrence_2007_2008.txt',
            'a', encoding='utf-8') as ff:
        # stop in 1966
        sql = '''SELECT t1.PMID, DescriptorName_UI, t2.publicationyear
FROM A06_MeshHeadingList t1
         inner join A01_Articles_publicationyear t2 on t1.PMID = t2.PMID
where toInt32(t2.publicationyear) between 2007 and 2008
order by t2.publicationyear ASC;'''
        data = click_server(sql)
        for PMID in data['PMID'].drop_duplicates(keep='first'):
            comb = data[data['PMID'] == PMID]['DescriptorName_UI']
            comb = comb.sort_values(0, ascending=True)
            mesh_comb = combinations(comb, 2)
            year = data[data['PMID'] == PMID]['publicationyear'].drop_duplicates(keep='first').values[0]
            for DescriptorName_UI_c in mesh_comb:
                print(str(PMID) + '|' + str(DescriptorName_UI_c[0]).strip() + '|' + str(
                    DescriptorName_UI_c[1]).strip() + '|' + str(year))
                ff.write(str(PMID) + '|' + str(DescriptorName_UI_c[0]).strip() + '|' + str(
                    DescriptorName_UI_c[1]).strip() + '|' + str(year) + '\n')


def cooperation_strength(node):
    pass


def evolution_mesh_ego_structure_analysis(year, i):
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_ego_structure_analysis_2.txt',
              'a') as ff:
        sql = '''
                SELECT t1.DescriptorName, year, type FROM V02_evolution_types t1 where toInt32(t1.year) >=2004
                  '''
        mesh = click_server(sql)
        print(mesh)
        mesh_names = mesh[mesh['year'] == str(year)].loc[:, ['DescriptorName', 'type']]
        print(mesh_names)
        for yr in range(year - 5, year):
            g = nx_network(yr)
            all_nodes = g.nodes()
            for mesh_name, type in mesh_names.values:
                # print(all_nodes)
                if mesh_name in all_nodes:
                    ego = nx.ego_graph(g, str(mesh_name))
                    ego_number = len(ego.nodes())
                    gg = ig.Graph.from_networkx(ego)
                    # print(ego_number)
                    # node_id = list(gg.vs['_nx_name']).index(mesh_name)
                    density = gg.density()
                    # # short_path = gg.shortest_paths()
                    # closeness = gg.closeness()
                    H2 = gg.triad_census().t201
                    H3 = gg.triad_census().t300
                    print(density, H2, H3)
                    # cliq_3 = len(gg.cliques(3, 3))
                    # print(cliq_3)
                    # print(density)
                    # ecent = np.mean(gg.evcent(directed=False, weights=gg.es["weight"]))
                    # print(ecent)
                    # shortest_path_length = len(gg.shortest_paths())
                    print(mesh_name, year, type, density, H2, H3, ego_number, yr)
                    print('Thread.....' + str(i))
                    ff.write(str(mesh_name) + '|' + str(year) + '|' + str(type) + '|' + str(density) + '|' + str(
                        H2) + '|' + str(H3) + '|' + str(ego_number) + '|' + str(yr) + '\n')
                else:
                    print(mesh_name, year, type, 'NAN', 'NAN', 'NAN', 'NAN', yr)
                    print('Thread.....' + str(i))
                    ff.write(str(mesh_name) + '|' + str(year) + '|' + str(
                        type) + '|' + 'NAN' + '|' + 'NAN' + '|' + 'NAN' + '|' + 'NAN' + '|' + str(yr) + '\n')


def evolution_mesh_structure_analysis():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_structure_analysis_weight.txt',
              'a') as f:
        sql = '''
            SELECT t1.DescriptorName, year, type FROM V02_evolution_types t1 where toInt32(t1.year) >=2004
              '''
        mesh = click_server(sql)
        print(mesh)
        for year in range(2004, 2017):
            g = knowledge_graph(year - 1)
            mesh_names = mesh[mesh['year'] == str(year)].loc[:, ['DescriptorName', 'type']]
            print(mesh_names)
            all_nodes = g.vs['name']
            all_number = len(all_nodes)
            for mesh_name, type in mesh_names.values:
                if mesh_name in all_nodes:
                    # deg = g.vs.find(mesh_name).closeness()
                    degree = g.vs.find(mesh_name).degree()
                    # print(avg_degree)
                    closeness = g.vs.find(mesh_name).closeness()
                    # betweenness = g.vs.find(mesh_name).betweenness()
                    # print(closeness)
                    clustering = g.transitivity_local_undirected(mesh_name, weights=g.es["weight"])
                    pagerank = g.vs.find(mesh_name).pagerank()
                    print(mesh_name, year, type, degree, closeness, clustering, pagerank)
                    f.write(str(mesh_name) + '|' + str(year) + '|' + str(type) + '|' + str(degree) + '|' + str(
                        closeness) + '|' + str(clustering) + '|' + str(pagerank) + '\n')
                else:
                    print(mesh_name, year, type, 'NAN', 'NAN', 'NAN', 'NAN')
                    f.write(str(mesh_name) + '|' + str(year) + '|' + str(
                        type) + '|' + 'NAN' + '|' + 'NAN' + '|' + 'NAN' + '|' + 'NAN' + '\n')


def multi_thread():
    start = 2003
    for i in range(1, 14):
        t = threading.Thread(target=evolution_mesh_ego_structure_analysis, args=(start + i, i))
        t.start()

        # evolution_mesh_ego_structure_analysis(2004)


if __name__ == '__main__':
    nx_network()
    # evolution_dynamic_structure_analysis()
    # DescriptorName_UI_occurrence()
    # evolution_mesh_ego_structure_analysis()
    # multi_thread()
    # evolution_mesh_structure_analysis()
