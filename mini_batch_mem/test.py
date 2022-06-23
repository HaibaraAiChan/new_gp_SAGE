from itertools import combinations
import networkx as nx
from networkx.generators.random_graphs import gnp_random_graph
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np

shared_neighbors={}
g= gnp_random_graph(4, 0.2)  # get a graph
g=nx.Graph()
g.add_node(0)
g.add_node(1)
g.add_node(2)
g.add_node(3)
g.add_edge(3,2)
g.add_edge(0,2)
g.add_edge(1,2)
g.add_edge(1,3)
fig = plt.figure(figsize=(5,5))
ax = plt.subplot(111)
ax.set_title('Graph - Shapes', fontsize=10)

nx.draw(g,pos=nx.spring_layout(g),with_labels = True)
plt.draw()
print(g)
plt.savefig("Graph.png", format="PNG")
neighbors = {node: set(g.neighbors(node)) for node in g.nodes}
for node_1, node_2 in combinations(g.nodes, r=2):
    shared_neighbors[(node_1, node_2)] = (
        len(neighbors[node_1] & neighbors[node_2])
    )
    print(shared_neighbors)
A = nx.to_scipy_sparse_matrix(g)
# print(A)
A = (A**2).todok()
# I = sp.sparse.eye(4, 4, 1)
# A=A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)
print(A)
# # A=A-I
# print(A[(0,1)])
# print(A[(0,2)])
# print(A[(0,3)])
# print(A[(1,2)])
# print(A[(1,3)])
# print(A[(2,3)])

# G = nx.DiGraph()
# G.add_edge("x", "a", capacity=3.0)
# G.add_edge("x", "b", capacity=1.0)
# G.add_edge("a", "c", capacity=3.0)
# G.add_edge("b", "c", capacity=5.0)
# G.add_edge("b", "d", capacity=4.0)
# G.add_edge("d", "e", capacity=2.0)
# G.add_edge("c", "y", capacity=2.0)
# G.add_edge("e", "y", capacity=3.0)

    

# fig = plt.figure(figsize=(5,5))
# ax = plt.subplot(111)
# ax.set_title('Graph - Shapes', fontsize=10)
# pos=nx.spring_layout(G)
# nx.draw(G,pos,with_labels = True)

# # Create edge labels
# labels = {e: str(G.edges[e]['capacity']) for e in G.edges}
# print(labels)
# # Draw edge labels according to node positions
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# # plt.show()
# plt.savefig("Graph_di.png", format="PNG")


# cut_value, partition = nx.minimum_cut(G, "x", "y")
# reachable, non_reachable = partition

# # print(cut_value)
# # print(partition)


# cutset = set()
# for u, nbrs in ((n, G[n]) for n in reachable):
#     cutset.update((u, v) for v in nbrs if v in non_reachable)
# # print(sorted(cutset))

# plt.clf()
# from networkx.algorithms.flow import shortest_augmenting_path
# G = nx.DiGraph()
# G.add_edge("x", "a", capacity=3.0)
# G.add_edge("x", "b", capacity=1.0)
# G.add_edge("a", "c", capacity=3.0)
# G.add_edge("b", "c", capacity=5.0)
# G.add_edge("b", "d", capacity=4.0)
# G.add_edge("d", "e", capacity=2.0)
# G.add_edge("c", "y", capacity=2.0)
# G.add_edge("e", "y", capacity=3.0)

# R = shortest_augmenting_path(G, "x", "y")

# fig = plt.figure(figsize=(6,5))
# ax = plt.subplot(111)
# ax.set_title('Graph - Shapes', fontsize=10)
# # pos=nx.spring_layout(G)
# nx.draw(R,pos,with_labels = True)

# # Create edge labels
# labels_2 = {e: str(R.edges[e]['capacity']) for e in R.edges}
# print(labels_2)

# # Draw edge labels according to node positions
# nx.draw_networkx_edge_labels(R, pos, edge_labels=labels_2,label_pos=0.2)
# plt.show()
# plt.savefig("Graph_r.png", format="PNG")
# # cut_set=nx.minimum_edge_cut(G, "x", "y")
# # print(cut_set)


# # # [('c', 'y'), ('x', 'b')]
# # cut_value == sum(G.edges[u, v]["capacity"] for (u, v) in cutset)



# # # import networkx as nx
# # import pandas as pd
# # import matplotlib.pyplot as plt


# # feature_1 = ['Boston', 'Boston', 'Chicago', 'ATX', 'NYC']
# # feature_2 = ['LA', 'SFO', 'LA', 'ATX', 'NJ']
# # score = ['1.00', '0.83', '0.34', '0.98', '0.89']

# # df = pd.DataFrame({'f1': feature_1, 'f2': feature_2, 'score': score})
# # print(df)

# # G = nx.from_pandas_edgelist(df=df, source='f1', target='f2', edge_attr='score')
# # pos = nx.spring_layout(G, k=10)  # For better example looking
# # nx.draw(G, pos, with_labels=True)
# # labels = {e: G.edges[e]['score'] for e in G.edges}
# # print(labels)
# # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# # plt.show()
# # plt.savefig("Graph_test.png", format="PNG")