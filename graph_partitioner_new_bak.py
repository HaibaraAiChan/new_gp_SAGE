import numpy 
import dgl
from numpy.core.numeric import Infinity
import multiprocessing as mp
import torch
import time
from statistics import mean
from my_utils import *
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import cupy as cp
import sys


# sys.path.insert(0,'..')
# from draw_graph import draw_dataloader_blocks_pyvis_total, gen_pyvis_graph_local

class Graph_Partitioner:
	def __init__(self, layer_block, args):
		self.balanced_init_ratio=args.balanced_init_ratio
		self.dataset=args.dataset
		self.layer_block=layer_block # local graph with global nodes indices
		self.local=False
		self.output_nids=layer_block.dstdata['_ID'] # tensor type
		self.local_output_nids=[]
		self.local_src_nids=[]
		self.src_nids_list= layer_block.srcdata['_ID'].tolist()
		self.full_src_len=len(layer_block.srcdata['_ID'])
		self.global_batched_seeds_list=[]
		self.local_batched_seeds_list=[]
		self.weights_list=[]
		self.alpha=args.alpha 
		self.walkterm=args.walkterm
		self.num_batch=args.num_batch
		self.selection_method=args.selection_method
		self.batch_size=0
		self.ideal_partition_size=0

		self.bit_dict={}
		self.side=0
		self.partition_nodes_list=[]
		self.partition_len_list=[]

		self.time_dict={}
		self.red_before=[]
		self.red_after=[]
		self.args=args
		
		
	def gen_batched_seeds_list(self):
		'''
		Parameters
		----------
		OUTPUT_NID: final layer output nodes id (tensor)
		selection_method: the graph partition method

		Returns
		-------
		'''
	
		full_len = len(self.local_output_nids)  # get the total number of output nodes
		self.batch_size=get_mini_batch_size(full_len,self.num_batch)
		
		indices=[]
		if self.selection_method == 'range_init_graph_partition' :
			t=time.time()
			indices = [i for i in range(full_len)]
			batches_nid_list, weights_list=gen_batch_output_list(self.local_output_nids,indices,self.batch_size)
			print('range_init for graph_partition spend: ', time.time()-t)
		elif self.selection_method == 'random_init_graph_partition' :
			t=time.time()
			indices = random_shuffle(full_len)
			batches_nid_list, weights_list=gen_batch_output_list(self.local_output_nids,indices,self.batch_size)
			print('random_init for graph_partition spend: ', time.time()-t)
		elif self.selection_method == 'balanced_init_graph_partition' :
			t=time.time()
			batches_nid_list, weights_list=self.balanced_init()
			print('balanced_init for graph_partition spend: ', time.time()-t)

			
		else:# selection_method == 'shared_neighbor_graph_partition':
			indices = torch.tensor(range(full_len)) #----------------------------TO DO
		
		# batches_nid_list, weights_list=gen_batch_output_list(self.output_nids,indices,self.batch_size)

		self.local_batched_seeds_list=batches_nid_list
		self.weights_list=weights_list

		print('The batched output nid list before graph partition')
		# print_len_of_batched_seeds_list(batches_nid_list)

		return 



	def remove_non_output_nodes(self):
		import copy
		local_src=copy.deepcopy(self.local_src_nids)
		mask_array = np.full(len(local_src),True, dtype=np.bool)
		mask_array[self.local_output_nids] = False
		from itertools import compress
		to_remove=list(compress(local_src, mask_array))
		return to_remove


	def get_src(self, seeds):
		in_ids=list(self.layer_block.in_edges(seeds))[0].tolist()
		src= list(set(in_ids+seeds))
		return src

	# def weighted_graph_bipart(self):
	# 	t1=time.time()
	# 	u, v =self.layer_block.edges()[0], self.layer_block.edges()[1] # local edges
	# 	g = dgl.graph((u,v))
	# 	print('g = dgl.graph((u,v))  spent ', time.time()-t1 )
	# 	t12 = time.time()
	# 	nx_g = dgl.to_networkx(g)  # local graph
	# 	print('nx_g = dgl.to_networkx(g)   spent ', time.time()-t12 )
	# 	t13 = time.time()
	# 	A = nx.to_scipy_sparse_matrix(nx_g)
	# 	print('A = nx.to_scipy_sparse_matrix(nx_g)   spent ', time.time()-t13 )
	# 	t14 = time.time()
	# 	AT= A.transpose()
	# 	# shared_neighbor_matrix = (AT*A)
	# 	from scipy.sparse import csc_matrix
	# 	shared_neighbor_matrix = AT.multiply(A)
	# 	# shared_neighbor_matrix = np.matmul(AT, A)
	# 	print('shared_neighbor_matrix = (AT*A)   spent ', time.time()-t14 )
	# 	t15 = time.time()
	# 	shared_neighbor_matrix=shared_neighbor_matrix.todok()
	# 	print('shared_neighbor_matrix = (AT*A).todok()   spent ', time.time()-t15 )
	# 	t16 = time.time()
	# 	shared_neighbor_matrix.setdiag(0)
	# 	print('shared_neighbor_matrix.setdiag(0), achieve weights  spent ', time.time()-t16 )
	# 	t2 = time.time()
		
	# 	# shared_neighbor_matrix.eliminate_zeros()
	# 	# tt=set(u.tolist()+v.tolist())
	# 	# remove = [node for node in tt if node not in self.local_output_nids]
	# 	remove = self.remove_non_output_nodes() # use mask to replace the for loop to speed up
	# 	print('get remove nodes spent ', time.time()-t2 )
	# 	print('remove nodes length')
	# 	print(len(remove))
	# 	t3 = time.time()
	# 	print(shared_neighbor_matrix)

	# 	# now we got shared neighbor numbers
	# 	# then we construct auxiliary graph on output nodes for output nodes (batches) partition
	# 	# {(a, b): 2} means 'a and b share two neighbors'
		
		
	# 	v=shared_neighbor_matrix.values()
	# 	k=shared_neighbor_matrix.keys()
	# 	nodes_pairs = list(shared_neighbor_matrix.keys())
	# 	print('prepare for output_shared_matrix spent ', time.time()-t3 )
	# 	print(len(k))
	# 	# t4 = time.time()
	# 	# output_shared_matrix = shared_neighbor_matrix
	# 	output_shared_matrix = {key: value  for key,value in zip(k,v) if (key[1] not in remove )and (key[0] not in remove) }
	# 	# print('get output_shared_matrix spent ', time.time()-t4 )
	# 	t5 = time.time()
	# 	#output_shared_matrix=shared_neighbor_matrix.fromkeys(nodes_pairs)
	# 	capacity = list(output_shared_matrix.values())
	# 	o_nodes_pairs = list(output_shared_matrix.keys())
	# 	start=(list(zip(*o_nodes_pairs))[0])
	# 	end=(list(zip(*o_nodes_pairs))[1])
	# 	df=pd.DataFrame({'source':start,'target':end, 'weight': capacity,'capacity':capacity})	
	# 	# df['src']=start
	# 	# df['dst']=end
	# 	# df['capacity']=capacity
	# 	print(df)
	# 	# df=pd.DataFrame((start, end, capacity),columns=['src','dst','capacity'])
	# 	G=nx.from_pandas_edgelist(df,edge_attr=True)
	# 	print('get auxiliary matrix spent ', time.time()-t5 )
	# 	t6 = time.time()
	# 	# print(G)
	# 	all_nodes=set(start+end)
	# 	length=len(set(start+end))
	# 	x=start[int(length/2)]
	# 	x=min(min(start),min(end))
	# 	y=max(max(start),max(end))
		
	# 	cut_value, partition = nx.minimum_cut(G, x, y)
	# 	print('cut_value')
	# 	print(cut_value)
	# 	print(partition) # local nid------------
	# 	part1=list(partition[0])
	# 	part2=list(partition[1])
	# 	print('get mini cut partition spent ', time.time()-t6 )
	# 	t7 = time.time()
	# 	#========================================= adjustment start =============================================
	# 	# If the output nodes are not included in partitions, we choose the closer partition to add.
	# 	T=set(self.local_output_nids)
	# 	P1=set(part1)
	# 	P2=set(part2)

	# 	if len(P1 | P2)<len( T ):
	# 		D = T.difference((P1 |P2))
	# 		for nid in list(D):
	# 			if nid in self.get_src(part1) and len(part1)<len(part2):
	# 				part1.append(nid)
	# 			elif nid in self.get_src(part2):
	# 				part2.append(nid)
	# 	# after this operation, if there still have output nodes left, append to the partition 2 directly.
	# 	P1=set(part1)
	# 	P2=set(part2)
	# 	if len(P1 | P2)<len( T ):
	# 		D = T.difference((P1 |P2))
	# 		if len(part1)>len(part2):
	# 			part2 = part2 + list(D)
	# 		else:
	# 			part1 = part1 + list(D)
	# 	#============================================= adjustment  end ===============================================
	# 	print('partition adjustment spent ', time.time()-t7 )
		
	# 	self.local_batched_seeds_list=[sorted(part1),sorted(part2)]
		
	# 	return 

	# def get_src(self,seeds):
	# 	in_ids=list(self.layer_block.in_edges(seeds))[0].tolist()
	# 	src= list(set(in_ids+seeds))
	# 	return src
	def weighted_graph_bipart(self):
		t1=time.time()
		u, v =self.layer_block.edges()[0], self.layer_block.edges()[1] # local edges
		g = dgl.graph((u,v))
		print('g = dgl.graph((u,v))  spent ', time.time()-t1 )
		# t12 = time.time()
		# nx_g = dgl.to_networkx(g)  # local graph
		# print('nx_g = dgl.to_networkx(g)   spent ', time.time()-t12 )
		t13 = time.time()
		A = g.adjacency_matrix()
		print('A = g.adjacency_matrix()   spent ', time.time()-t13 )
		t14 = time.time()
		AT= torch.transpose(A, 0, 1)
		print(A)
		print(AT)
		m_at = AT._indices().tolist()
		m_a  = A._indices().tolist()
		length = len(m_a[0])
		print(m_at)
		print(m_a)
		g_at = dgl.graph((m_at[0], m_at[1]))
		g_at.edata['w'] = torch.ones(length).requires_grad_()
		g_a = dgl.graph((m_a[0], m_a[1]))
		g_a.edata['w'] = torch.ones(length).requires_grad_()
		auxiliary_graph = dgl.adj_product_graph(g_at, g_a, 'w')
		t2 = time.time()
		remove = self.remove_non_output_nodes() # use mask to replace the for loop to speed up
		print('get remove nodes spent ', time.time()-t2 )
		print('remove nodes length')
		print(len(remove))
		auxiliary_graph.remove_nodes(torch.tensor(remove))

		partition = dgl.metis_partition(g=auxiliary_graph,k=2)
		print(partition)
		print(partition[0].ndata[dgl.NID])
		print(partition[1].ndata[dgl.NID])
		part0=partition[0].ndata[dgl.NID].tolist()
		part1=partition[1].ndata[dgl.NID].tolist()
		# shared_neighbor_matrix = torch.sparse.mm(AT, A.to_dense()) # torch only support sparse*dense matrix multiplication
		# from scipy.sparse import csc_matrix
		# shared_neighbor_matrix = AT.multiply(A)
		# shared_neighbor_matrix = np.matmul(AT, A)
		# print('shared_neighbor_matrix = (AT.A)   spent ', time.time()-t14 )
		# t15 = time.time()
		# shared_neighbor_matrix=shared_neighbor_matrix.todok()
		# print('shared_neighbor_matrix = (AT*A).todok()   spent ', time.time()-t15 )
		# t16 = time.time()
		# shared_neighbor_matrix.fill_diagonal_(0)
		# shared_neighbor_matrix.setdiag(0)
		# print('shared_neighbor_matrix set diagonal zeros, achieve weights spent ', time.time()-t16 )
		# t2 = time.time()
		
		# shared_neighbor_matrix.eliminate_zeros()
		# tt=set(u.tolist()+v.tolist())
		# remove = [node for node in tt if node not in self.local_output_nids]
		# remove = self.remove_non_output_nodes() # use mask to replace the for loop to speed up
		# print('get remove nodes spent ', time.time()-t2 )
		# print('remove nodes length')
		# print(len(remove))
		# t3 = time.time()
		# print('dense: '+str(shared_neighbor_matrix))
		# print(type(shared_neighbor_matrix)) 
		# shared_neighbor_matrix = shared_neighbor_matrix.to_sparse()
		# print('sparse: '+str(shared_neighbor_matrix))
		# now we got shared neighbor numbers
		# then we construct auxiliary graph on output nodes for output nodes (batches) partition
		# {(a, b): 2} means 'a and b share two neighbors'
		
		
		# v=shared_neighbor_matrix.values()
		# k=shared_neighbor_matrix.keys()
		# nodes_pairs = list(shared_neighbor_matrix.keys())
		# print('prepare for output_shared_matrix spent ', time.time()-t3 )
		# print(len(k))
		# t4 = time.time()
		# output_shared_matrix = shared_neighbor_matrix
		# output_shared_matrix = {key: value  for key,value in zip(k,v) if (key[1] not in remove )and (key[0] not in remove) }
		# print('get output_shared_matrix spent ', time.time()-t4 )
		# t5 = time.time()
		#output_shared_matrix=shared_neighbor_matrix.fromkeys(nodes_pairs)
		# capacity = list(output_shared_matrix.values())
		# o_nodes_pairs = list(output_shared_matrix.keys())
		# start=(list(zip(*o_nodes_pairs))[0])
		# end=(list(zip(*o_nodes_pairs))[1])
		# df=pd.DataFrame({'source':start,'target':end, 'weight': capacity,'capacity':capacity})	
		# # df['src']=start
		# # df['dst']=end
		# # df['capacity']=capacity
		# print(df)
		# # df=pd.DataFrame((start, end, capacity),columns=['src','dst','capacity'])
		# G=nx.from_pandas_edgelist(df,edge_attr=True)
		# t5 = time.time()
		# e_weight = shared_neighbor_matrix._values()
		# indices  = shared_neighbor_matrix._indices()

		# print('get auxiliary matrix spent ', time.time()-t5 )
		# t6 = time.time()
		# # print(G)
		# all_nodes=set(start+end)
		# length=len(set(start+end))
		# x=start[int(length/2)]
		# x=min(min(start),min(end))
		# y=max(max(start),max(end))
		
		# cut_value, partition = nx.minimum_cut(G, x, y)
		# print('cut_value')
		# print(cut_value)
		# print(partition) # local nid------------
		# part1=list(partition[0])
		# part2=list(partition[1])
		# print('get mini cut partition spent ', time.time()-t6 )
		# t7 = time.time()
		#========================================= adjustment start =============================================
		# If the output nodes are not included in partitions, we choose the closer partition to add.
		# T=set(self.local_output_nids)
		# P1=set(part1)
		# P2=set(part2)

		# if len(P1 | P2)<len( T ):
		# 	D = T.difference((P1 |P2))
		# 	for nid in list(D):
		# 		if nid in self.get_src(part1) and len(part1)<len(part2):
		# 			part1.append(nid)
		# 		elif nid in self.get_src(part2):
		# 			part2.append(nid)
		# # after this operation, if there still have output nodes left, append to the partition 2 directly.
		# P1=set(part1)
		# P2=set(part2)
		# if len(P1 | P2)<len( T ):
		# 	D = T.difference((P1 |P2))
		# 	if len(part1)>len(part2):
		# 		part2 = part2 + list(D)
		# 	else:
		# 		part1 = part1 + list(D)
		#============================================= adjustment  end ===============================================
		# print('partition adjustment spent ', time.time()-t7 )
		
		self.local_batched_seeds_list=[sorted(part0),sorted(part1)]
		
		return 

		# torch.transpose(x, 0, 1)








	def get_src_len(self,seeds):
		in_ids=list(self.layer_block.in_edges(seeds))[0].tolist()
		src_len= len(list(set(in_ids+seeds)))
		return src_len



	def get_partition_src_len_list(self):
		partition_src_len_list=[]
		for seeds_nids in self.local_batched_seeds_list:
			partition_src_len_list.append(self.get_src_len(seeds_nids))
		
		self.partition_src_len_list=partition_src_len_list
		return partition_src_len_list


	def graph_partition(self):
		
		full_batch_subgraph=self.layer_block #heterogeneous graph (block)
		self.bit_dict={}
		print('----------------------------  graph partition start---------------------')
		# if num_batch == 0:
		#     self.output_nids/
		self.ideal_partition_size=(self.full_src_len/self.num_batch)
		if self.num_batch==2:
			t2=time.time()
			self.weighted_graph_bipart()
			print('self.weighted_graph_bipart() spend ', time.time()-t2 )
		# src_ids=list(full_batch_subgraph.edges())[0]
		# dst_ids=list(full_batch_subgraph.edges())[1]
		# local_g = dgl.graph((src_ids, dst_ids)) #homogeneous graph
		# local_g = dgl.remove_self_loop(local_g)
		# # from draw_graph import draw_graph
		# # draw_graph(local_g)
		# self.layer_block=local_g
		# global block_to_graph
		# block_to_graph=local_g

		# self.gen_batched_seeds_list() # based on user choice to split output nodes 

		# src_len_list= self.get_partition_src_len_list()


		weight_list=get_weight_list(self.local_batched_seeds_list)
		src_len_list=self.get_partition_src_len_list()
		print('after graph partition')
		self.weights_list=weight_list
		self.partition_len_list=src_len_list

		return self.local_batched_seeds_list, weight_list, src_len_list
	
	def global_to_local(self):
		
		sub_in_nids = self.src_nids_list
		print('src global')
		print(sub_in_nids)#----------------
		global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
		self.local_output_nids = list(map(global_nid_2_local.get, self.output_nids.tolist()))
		print('dst local')
		print(self.local_output_nids)#----------------
		self.local_src_nids = list(map(global_nid_2_local.get, self.src_nids_list))
		
		self.local=True
		return 	


	def local_to_global(self):
		sub_in_nids = self.src_nids_list
		local_nid_2_global = { i: sub_in_nids[i] for i in range(0, len(sub_in_nids))}
		
		global_batched_seeds_list=[]
		for local_in_nids in self.local_batched_seeds_list:
			global_in_nids = list(map(local_nid_2_global.get, local_in_nids))
			global_batched_seeds_list.append(global_in_nids)

		self.global_batched_seeds_list=global_batched_seeds_list
		# print('-----------------------------------------------global batched output nodes id----------------------------')
		for inp in self.global_batched_seeds_list:
			
			print(sorted(inp))
		# print(self.global_batched_seeds_list)
		self.local=False
		return 


	def init_graph_partition(self):
		ts = time.time()
		
		self.global_to_local() # global to local            self.local_batched_seeds_list
		print('global_2_local spend time (sec)', (time.time()-ts))
		
		# t1 = time.time()
		# self.gen_batched_seeds_list()

		t2=time.time()
		# Then, the graph_parition is run in block to graph local nids,it has no relationship with raw graph
		self.graph_partition()
		print('graph partition algorithm spend time', time.time()-t2)
		# after that, we transfer the nids of batched output nodes from local to global.
		self.local_to_global() # local to global         self.global_batched_seeds_list
		t_total=time.time()-ts

		return self.global_batched_seeds_list, self.weights_list, t_total, self.partition_len_list