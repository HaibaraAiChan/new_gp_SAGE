from collections import defaultdict
import torch
import numpy
# converts from adjacency matrix coo (undircted)to adjacency list
def convert(a):
    
    adjList = defaultdict(list)
    for i in range(len(a)):
        flag = False
        for j in range(len(a[i])):
            # if a[i][j]== 0:
                # print('there is no edge')
            if a[i][j]== 1:
                adjList[i].append(j)
                flag = True
                # print('i,j = '+ str(i)+','+str(j))
        if not flag:
            # print(str(i)+ ' is isolated')
            adjList[i].append("")
        
    # print(adjList)
    return adjList
 
# driver code
# a =[[0, 0, 1], [0, 0, 1], [1, 1, 0]] # adjacency matrix
a = torch.load('CSR_matrix.pt')
# a = torch.load('CSR_matrix_arxiv.pt')
# w = torch.load('counter-weights.pt')
# edges_weight = torch.load('edges-weights.pt')
# edges = torch.load('edges.pt')
# print(a.shape.data[0])
# print(a)

# print(w)
# print(edges)

AdjList = convert(a)
# print("% Adjacency List format")
print(a.shape[0], end =" ") # nodes number 
# print(a.sparse_dim(),end =" ")
# print(a.dense_dim(),end ="")
# print(a.coalesce().indices(),end =" ")
print(list(a.coalesce().values().size())[0]//2,end ="")
print()
# # print the adjacency list
# for i in AdjList:
#     # print(i, end ="")
#     for j in AdjList[i]:
#         # print(" -->  {}".format(j), end ="")
#         if j != "":
#             print("{} ".format(j+1), end ="")
#         # the node id should satrt from 1 instead of 0, we add 1 for all node id, later we need change it back after partition
#     print()
