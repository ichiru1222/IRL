import networkx as nx
import math
import random
import numpy as np
import matplotlib.pyplot as plt

n = 40
p = 0.5

class Graphenv:

 

    def __init__(self, graph, reward, random_error = None):
        #e.g.) shape = [4,4] , reward = np.zeos(nS)

        
        self.nS = nx.number_of_nodes(graph) #状態数　（ノードの数）
        print(self.nS)
        self.nA = [x[1] for x in graph.degree()] #行動数　各状態に対しての行動数を辞書に格納
        self.graph = graph
        self.P = self.make_prob()


    def make_prob(self):
        """
        (ex:output) 横：状態数　縦：最大行動数 pは確定的
                    [[[0. 0. 1. 0. 0.]
                    [0. 0. 0. 1. 0.]
                    [0. 0. 0. 0. 0.]]

                    [[0. 0. 1. 0. 0.]
                    [0. 0. 0. 0. 1.]
                    [0. 0. 0. 0. 0.]]

                    [[1. 0. 0. 0. 0.]
                    [0. 1. 0. 0. 0.]
                    [0. 0. 0. 1. 0.]]

                    [[1. 0. 0. 0. 0.]
                    [0. 0. 1. 0. 0.]
                    [0. 0. 0. 0. 1.]]

                    [[0. 1. 0. 0. 0.]
                    [0. 0. 0. 1. 0.]
                    [0. 0. 0. 0. 0.]]]
        """
        Alen = max([x[1] for x in self.graph.degree()])
        P = np.array([[list(np.identity(self.nS)[i]) for i in [x[1] for x in self.graph.edges([e])]] + [[0] * self.nS] * (Alen - len(self.graph.edges([e]))) for e in range(self.nS)], dtype=float)

        return P



def make_random_graph(number_of_nodes, p): #number_of_nodes:ノードの数，p:ノード間にエッジを生成する確率
    while True:
        graph = nx.fast_gnp_random_graph(number_of_nodes, p)
        if nx.is_connected(graph):
            break
    return graph


def make_expart_paths(graph, number_of_exparts):  #エキスパート軌跡の生成 maxentは軌跡の状態数をそろえる必要がある
    len_path = dict(nx.all_pairs_dijkstra(graph))
    all_shortest_paths = {} #あるノードから別のノードまでの最短経路を格納　{0:{0:[0],1:[0,1]},1:{2:[1,3,2]}} 辞書の中に辞書
    for i in range(nx.number_of_nodes(graph)):
        all_shortest_paths[i] = len_path[i][1]
    
    #ノードをランダムに二つ選択し，それらの最短経路を取り出す
    exparts_paths = []
    for i in range(number_of_exparts):
        first_node = random.randint(0, nx.number_of_nodes(graph)-1)
        while True:
            second_node = random.randint(0, nx.number_of_nodes(graph)-1)
            if second_node != first_node:
                break
        exparts_paths.append(all_shortest_paths[first_node][second_node])
    
    len_path_list = [len(x) for x in exparts_paths]
    max_path = max(len_path_list)

    exparts_paths_eq = []

    for path in exparts_paths:
        if len(path) != max_path:
            for i in range(max_path - len(path)):
                path.append(path[-1])
            exparts_paths_eq.append(path)
        else:
            exparts_paths_eq.append(path)



    
    return exparts_paths_eq

def spacesyntax(graph):
    closeness=nx.closeness_centrality(graph) #TDの導出、辞書型
    mds = {}
    ras = {}
    intVs = {}
    nodes_len = len(list(graph.nodes))
    #print(closeness)
    for key, value in closeness.items():
        mds[key] = 1/value
    #print(mds)
    for key, md in mds.items():
        ras[key] = 2 * (md - 1)/(nodes_len - 2)
    #print(ras)
    dnup = 2 * (nodes_len * (math.log2((nodes_len + 2) / 3) - 1) + 1)
    dndown = (nodes_len - 1) * (nodes_len - 2)
    dn = dnup/dndown
    #print(dn)
    for key, ra in ras.items():
        intVs[key] = dn/ra
    #print(intVs)

    return intVs
   



    
    
    








if __name__ == '__main__':
    number_of_nodes = 100
    p = 0.1
    number_of_exparts = 10
    reward = np.zeros(10)
    graph = make_random_graph(number_of_nodes, p)
    
   
    env = Graphenv(graph, reward)
    print(env.P)
    print(env.P.shape)
    print(env.P.dtype)

    print(make_expart_paths(graph,number_of_exparts))
    #print(spacesyntax(graph))

    #print(graph.)

    #print(graph)
    nx.draw(graph, with_labels=True)

    plt.show(graph)