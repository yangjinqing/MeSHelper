# -*- coding: UTF-8 -*-
import pandas as pd
import igraph as ig
import networkx as nx
from clickhouse_driver import Client as click_client
import re


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

# create a occurrence keyword network
def knowledge_graph(year):
    sql = '''
    SELECT t1.DescriptorName_start, t1.DescriptorName_end, count() as weight
FROM A15_keyword_pairs t1 where toInt32(t1.Year) <= ''' + str(year) + '''
group by t1.DescriptorName_start, t1.DescriptorName_end;
          '''
    data = click_server(sql)
    pairs = [tuple(x) for x in data.values]
    g = ig.Graph.TupleList(pairs, edge_attrs="weight", directed=False)
    print(ig.summary(g))
    return g

def nx_network(year):
    sql = '''
        SELECT t1.DescriptorName_start, t1.DescriptorName_end, count() as weight
    FROM A15_keyword_pairs t1 where toInt32(t1.Year) <= ''' + str(year) + '''
    group by t1.DescriptorName_start, t1.DescriptorName_end;
              '''
    data = click_server(sql)
    # print(refer_pairs)
    edges = data
    # print(tuple(edges.values))
    g = nx.Graph()
    g.add_weighted_edges_from(tuple(edges.values))
    print(g.size())
    return g


def evolution_mesh_structure_analysis():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_ego_structure_analysis.txt', 'a') as ff:
        sql = '''
            SELECT t1.DescriptorName, year, type FROM V02_evolution_types t1 where toInt32(t1.year) >=2004
              '''
        mesh = click_server(sql)
        print(mesh)
        for year in range(2004, 2017):
            g = knowledge_graph(year-1)
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
                    shortest_path = g.vs.find(mesh_name).shortest_paths_dijkstra()
                    # print(closeness)
                    clustering = g.transitivity_local_undirected(mesh_name, weights=g.es["weight"])
                    print(mesh_name, year, type, degree, closeness, shortest_path, clustering)

                else:
                    print(mesh_name, year, type, 'NAN', 'NAN', 'NAN', 'NAN')

def evolution_mesh_ego_structure_analysis():
    with open('/home/lab109/data/MeSH_2001_2020/mesh_evolution/knowledge_graph/mesh_ego_structure_analysis.txt', 'a') as ff:
        sql = '''
                SELECT t1.DescriptorName, year, type FROM V02_evolution_types t1 where toInt32(t1.year) >=2004
                  '''
        mesh = click_server(sql)
        print(mesh)
        for year in range(2004, 2017):
            mesh_names = mesh[mesh['year'] == str(year)].loc[:, ['DescriptorName', 'type']]
            print(mesh_names)
            g = nx_network(year - 1)
            all_nodes = g.nodes()
            for mesh_name, type in mesh_names.values:
                # print(all_nodes)
                if mesh_name in all_nodes:
                    ego = nx.ego_graph(g, str(mesh_name))
                    ego_number = len(ego.nodes())
                    gg = ig.Graph.from_networkx(ego)
                    # node_id = list(gg.vs['_nx_name']).index(mesh_name)
                    density = gg.density()
                    print(density)
                    print(mesh_name, year, type, density, ego_number)
                    ff.write(str(mesh_name)+'|'+str(year)+'|' + str(type)+'|'+str(density)+'|' +'|' + str(ego_number)+'\n')
                else:
                    print(mesh_name, year, type, 'NAN', 'NAN')
                    ff.write(str(mesh_name)+'|'+str(year)+'|' + str(type)+'|'+str('NAN')+'|' + str('NAN') +'\n')

if __name__ == '__main__':
    evolution_mesh_ego_structure_analysis()