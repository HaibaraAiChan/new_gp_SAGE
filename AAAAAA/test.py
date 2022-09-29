from collections import defaultdict
import torch
# converts from adjacency matrix coo (undircted)to adjacency list
def convert(a):
    adjList = defaultdict(list)
    for i in range(len(a)):
        for j in range(len(a[i])):
                       if a[i][j]== 1:
                           adjList[i].append(j)
    return adjList
 
# driver code
a =[[0, 0, 1], [0, 0, 1], [1, 1, 0]] # adjacency matrix
a = torch.load('CSR_matrix.pt')
# w = torch.load('counter-weights.pt')
# edges_weight = torch.load('edges-weights.pt')
# edges = torch.load('edges.pt')
print(a.shape)
print(a)

# print(w)
# print(edges)

AdjList = convert(a)
print("Adjacency List:")
# print the adjacency list
for i in AdjList:
    print(i, end ="")
    for j in AdjList[i]:
        print("   {}".format(j), end ="")
    print()