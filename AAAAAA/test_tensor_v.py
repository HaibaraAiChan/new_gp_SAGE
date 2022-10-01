import numpy as np
# converts from adjacency matrix coo (undircted)to adjacency list
def convert(nids, src_tensor, dst_tensor):

    # adjList = defaultdict(list)
    for i in range(nids):
        indices= ((src_tensor == i).nonzero(as_tuple=True)[0])
        # print(indices)
        if len(indices)==0:
            # adjList[i].append("")
            print()
        elif len(indices)>0:
            # adjList[i]=(list(dst_tensor[indices]))
            neighbors = list(dst_tensor[indices])
            for nei in neighbors:
                print("{} ".format(nei+1), end ="")
            print()
    # print(adjList)
    # return adjList
    return

# driver code
# a =[[0, 0, 1], [0, 0, 1], [1, 1, 0]] # adjacency matrix
# a = torch.load('CSR_matrix.pt')
a = torch.load('CSR_matrix_arxiv.pt')
# print(a)
# print(a.shape) 


# print(a.shape[0]) # nodes number 
# print(a.coalesce().indices()[0])
# src = list(list(a.coalesce().indices())[0])
src_tensor = a.coalesce().indices()[0]
dst_tensor = a.coalesce().indices()[1]
# print(src)
# dst = list(list(a.coalesce().indices())[1])
nids = list(a.shape)[0]

print(nids, end =" ")
print(list(a.coalesce().values().size())[0]//2,end ="")
print()
convert(nids, src_tensor, dst_tensor)

# # print(list(list(a.coalesce().indices())[0]),end =" ")
# # print(len(list(list(a.coalesce().indices())[0])),end =" ")

# # print the adjacency list
# for i in AdjList:
#     # print(i, end =" ")
#     for j in AdjList[i]:
#         # print(" -->  {}".format(j), end ="")
#         if j != "":
#             print("{} ".format(j+1), end ="")
#         # the node id should satrt from 1 instead of 0, we add 1 for all node id, later we need change it back after partition
#     print()
                                                                                                                                      32,1          Bot
