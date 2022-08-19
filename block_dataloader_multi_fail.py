import torch
import dgl
import numpy
import time
import pickle
import io
from math import ceil
from math import floor
from math import ceil
from itertools import islice
from statistics import mean
from multiprocessing import Manager, Pool
from multiprocessing import Process, Value, Array

from graph_partitioner_new import Graph_Partitioner
from draw_graph import draw_dataloader_blocks_pyvis

from my_utils import gen_batch_output_list
from memory_usage import see_memory_usage

from sortedcontainers import SortedList, SortedSet, SortedDict
from multiprocessing import Process, Queue
from collections import Counter, OrderedDict
import copy

class OrderedCounter(Counter, OrderedDict):
	'Counter that remembers the order elements are first encountered'

	def __repr__(self):
		return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

	def __reduce__(self):
		return self.__class__, (OrderedDict(self),)
#------------------------------------------------------------------------
# def unique_tensor_item(combined):
# 	uniques, counts = combined.unique(return_counts=True)
# 	return uniques.type(torch.long)




def get_global_graph_edges_ids_block(raw_graph, block):
	
	edges=block.edges(order='eid', form='all')
	edge_src_local = edges[0]
	edge_dst_local = edges[1]
	# edge_eid_local = edges[2]
	induced_src = block.srcdata[dgl.NID]
	induced_dst = block.dstdata[dgl.NID]
	induced_eid = block.edata[dgl.EID] 
		
	raw_src, raw_dst=induced_src[edge_src_local], induced_dst[edge_dst_local]
	# raw_src, raw_dst=induced_src[edge_src_local], induced_src[edge_dst_local]
	
	# in homo graph: raw_graph 
	global_graph_eids_raw = raw_graph.edge_ids(raw_src, raw_dst)
	# https://docs.dgl.ai/generated/dgl.DGLGraph.edge_ids.html?highlight=graph%20edge_ids#dgl.DGLGraph.edge_ids
	# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids

	return global_graph_eids_raw, (raw_src, raw_dst)



def generate_one_block(raw_graph, global_eids, global_srcnid, global_dstnid):
	'''

	Parameters
	----------
	G    global graph                     DGLGraph
	eids  cur_batch_subgraph_global eid   tensor int64

	Returns
	-------

	'''
	_graph = dgl.edge_subgraph(raw_graph, global_eids, store_ids=True)
	edge_dst_list = _graph.edges(order='eid')[1].tolist()
	dst_local_nid_list=list(OrderedCounter(edge_dst_list).keys())
	
	new_block = dgl.to_block(_graph, dst_nodes=torch.tensor(dst_local_nid_list, dtype=torch.long))
	new_block.srcdata[dgl.NID] = global_srcnid
	new_block.dstdata[dgl.NID] = global_dstnid
	new_block.edata['_ID']=_graph.edata['_ID']

	return new_block


def func(output_nid, current_layer_block, dict_nid_2_local ):
	# print("in:", msg)
	# time.sleep(3)
	print("start to do =======")
	local_output_nid = list(map(dict_nid_2_local.get, output_nid))
	print("start to do 2=======")
	local_in_edges_tensor = current_layer_block.in_edges(local_output_nid, form='all')
	print("start to do 3=======")
	mini_batch_src_local= list(local_in_edges_tensor)[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
	print("start to do 4=======")
	mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.
	print("start to do 5=======")
	mini_batch_dst_local= list(local_in_edges_tensor)[1]
	if set(mini_batch_dst_local.tolist()) != set(local_output_nid):
		print('local dst not match')
	eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
	global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  bottleneck  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	mini_batch_src_global = list(OrderedDict.fromkeys(mini_batch_src_global))
	c=OrderedCounter(mini_batch_src_global)
	list(map(c.__delitem__, filter(c.__contains__,output_nid)))
	r_=list(c.keys())
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$   bottleneck  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	src_nid = torch.tensor(output_nid + r_, dtype=torch.long)
	output_nid = torch.tensor(output_nid, dtype=torch.long)
	
	return (src_nid, output_nid, global_eid_tensor)

def log_result(src, output, eid):
	print(len(src))
	print(len(output))
	print(len(eid))
	print("Succesfully get callback! With result: ")


def check_connections_block(batched_nodes_list, current_layer_block):
	str_=''
	res=[]
	print('check_connections_block*********************************')

	# adj_tensor = current_layer_block.adj_sparse('coo')
	# print('adj tensor')
	# print(adj_tensor)
	# print()

	induced_src = current_layer_block.srcdata[dgl.NID]

	eids_global = current_layer_block.edata['_ID']

	t1=time.time()
	src_nid_list = induced_src.tolist()
	dict_nid_2_local = dict(zip(src_nid_list, range(len(src_nid_list)))) # speedup 
	str_+= 'time for parepare 1: '+str(time.time()-t1)+'\n'
	# t11=time.time()
	# dict_nid_2_local = {src_nid_list[i]: i for i in range(0, len(src_nid_list))}
	# str_+= 'time for parepare 2: '+str(time.time()-t11)+'\n'
	# import multiprocessing 
	# pool = multiprocessing.Pool(processes = 3)
	# from functools import partial
	# func_partial = partial(func, current_layer_block=current_layer_block, dict_nid_2_local=dict_nid_2_local)
	# from pathos.multiprocessing import ProcessingPool as Pool
	# pool = Pool(processes = 0)
	# for step, output_nid in enumerate(batched_nodes_list):
	# 	tmp = pool.map(func, output_nid, current_layer_block, dict_nid_2_local)
	# 	print(tmp)
	# 	res.append(pool.map(func, output_nid, current_layer_block, dict_nid_2_local))
	#####################################################################################
	print('pool *********************************')
	from multiprocessing import Pool
	from itertools import repeat
	zip_args = []
	tmp=[]
	cur_blocks = []
	dict_list = []
	import copy
	with Pool() as pool:
	# res = pool.starmap_async(func, batched_nodes_list, current_layer_block, dict_nid_2_local, callback = log_result).get()
		for step, output_nid in enumerate(batched_nodes_list):
			cur_blocks.append(copy.deepcopy(current_layer_block)) 
			dict_list.append(copy.deepcopy(dict_nid_2_local))
		zip_args=list(zip(batched_nodes_list, cur_blocks, dict_list)) 
		print('zip_args~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		# tmp = pool.starmap_async(func, zip_args)
		tmp = pool.starmap_async(func, zip(batched_nodes_list, repeat(current_layer_block), repeat(dict_nid_2_local)))
		print('tmp~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		for (src, output, eid) in tmp.get():
			res.append((src, output, eid))
		pool.close()
		pool.join()

	print('----------------------------------------------------------------pool starmap async res')
	print(res)
	return res
	#####################################################################################
	# print('parmap *********************************')
	# import parmap
	# zip_args = []
	# cur_blocks = []
	# dict_list = []
	# for step, output_nid in enumerate(batched_nodes_list):
	# 	cur_blocks.append(copy.deepcopy(current_layer_block)) 
	# 	dict_list.append(copy.deepcopy(dict_nid_2_local))
	# zip_args=list(zip(batched_nodes_list, cur_blocks, dict_list)) 
	# print('zip_args')
	# # print(zip_args)
	# tmp = parmap.starmap_async(func, zip_args)
	# for (src, output, eid) in tmp.get():
	# 	res.append((src, output, eid))
	# print('----------------------------------------------------------------pool starmap async res')
	# print(res)
	# return res
	#####################################################################################



	# for step, output_nid in enumerate(batched_nodes_list):
	# 	# in current layer subgraph, only has src and dst nodes,
	# 	# and src nodes includes dst nodes, src nodes equals dst nodes.
	# 	tt=time.time()
	# 	local_output_nid = list(map(dict_nid_2_local.get, output_nid))

	# 	str_+= 'local_output_nid generation: '+ str(time.time()-tt)+'\n'
	# 	tt1=time.time()

	# 	local_in_edges_tensor = current_layer_block.in_edges(local_output_nid, form='all')

	# 	str_+= 'local_in_edges_tensor generation: '+str(time.time()-tt1)+'\n'
	# 	tt2=time.time()
	# 	# return (𝑈,𝑉,𝐸𝐼𝐷)
	# 	# get local srcnid and dstnid from subgraph
	# 	mini_batch_src_local= list(local_in_edges_tensor)[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
	# 	mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.

	# 	str_+= 'mini_batch_src_global generation: '+str( time.time()-tt2) +'\n'
	# 	tt3=time.time()

	# 	mini_batch_dst_local= list(local_in_edges_tensor)[1]
	# 	# mini_batch_dst_global= induced_src[mini_batch_dst_local].tolist()
	# 	if set(mini_batch_dst_local.tolist()) != set(local_output_nid):
	# 		print('local dst not match')
	# 	eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
	# 	global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
	# # 	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  bottleneck  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	# 	str_+= "&&&&&&&&&&&&&&& before remove duplicate length: "+ str(len(mini_batch_src_global))+'\n'
	# 	ttp=time.time()
	# 	# print(mini_batch_src_global)
	# 	mini_batch_src_global = list(OrderedDict.fromkeys(mini_batch_src_global))
	# 	str_+= "&&&&&&&&&&&&&&& after remove duplicate length: "+ str(len(mini_batch_src_global)) +'\n\n'
	# 	ttp1=time.time()
	# 	# print(mini_batch_src_global)
	# 	c=OrderedCounter(mini_batch_src_global)
	# 	list(map(c.__delitem__, filter(c.__contains__,output_nid)))
	# 	r_=list(c.keys())
	# 	# 	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$   bottleneck  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	# 	str_+= 'remove duplicate: '+ str(ttp1-ttp)+'\n'
	# 	str_+= 'r_  generation: '+ str(time.time()-ttp1)+'\n'
		
		
	# 	# add_src=[i for i in mini_batch_src_global if i not in output_nid] 
	# 	# r_ =remove_duplicate(add_src) #---------------too slow
	
	# 	src_nid = torch.tensor(output_nid + r_, dtype=torch.long)
	# 	output_nid = torch.tensor(output_nid, dtype=torch.long)

	# 	res.append((src_nid, output_nid, global_eid_tensor))
	# 	print(str_)
	# return res


def check_connections_block_clear(batched_nodes_list, current_layer_block):
	res=[]
	induced_src = current_layer_block.srcdata[dgl.NID]
	eids_global = current_layer_block.edata['_ID']
	src_nid_list = induced_src.tolist()
	dict_nid_2_local = {src_nid_list[i]: i for i in range(0, len(src_nid_list))}
	
	for step, output_nid in enumerate(batched_nodes_list):
		# in current layer subgraph, only has src and dst nodes,
		local_output_nid = list(map(dict_nid_2_local.get, output_nid))
		local_in_edges_tensor = current_layer_block.in_edges(local_output_nid, form='all')
		# return (𝑈,𝑉,𝐸𝐼𝐷)
		# get local srcnid and dstnid from subgraph
		mini_batch_src_local= list(local_in_edges_tensor)[0] # local (𝑈,𝑉,𝐸𝐼𝐷) 𝑈 ;
		mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.
		
		eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷) 𝐸𝐼𝐷 ; 
		global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
		# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  bottleneck  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
		mini_batch_src_global = list(OrderedDict.fromkeys(mini_batch_src_global)) # remove duplicated 

		c=OrderedCounter(mini_batch_src_global)
		list(map(c.__delitem__, filter(c.__contains__,output_nid)))
		r_=list(c.keys())
		# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$   bottleneck  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
		src_nid = torch.tensor(output_nid + r_, dtype=torch.long)
		output_nid = torch.tensor(output_nid, dtype=torch.long)
		res.append((src_nid, output_nid, global_eid_tensor, local_output_nid))
	return res


def generate_blocks_for_one_layer_block(raw_graph, layer_block, batches_nid_list):
	# see_memory_usage("------------------------------- before        generate_blocks_for_one_layer_block ")
	
	# layer_eid = layer_block.edata[dgl.NID].tolist() # we have changed it to global eid
	# print(sorted(layer_eid))
	
	blocks = []
	check_connection_time = []
	block_generation_time = []

	t1= time.time()
	batches_temp_res_list = check_connections_block(batches_nid_list, layer_block)
	# return
	t2 = time.time()
	check_connection_time.append(t2-t1) #------------------------------------------
	print('----------------------check connections block total spend -----------------------------', t2-t1)
	src_list=[]
	dst_list=[]

	# see_memory_usage("------------------------------- before     for loop    batches_temp_res_list ")
	for step, (srcnid, dstnid, current_block_global_eid) in enumerate(batches_temp_res_list):
	# for step, (srcnid, dstnid, current_block_global_eid, src_e, dst_e) in enumerate(batches_temp_res_list):
		# print('batch ' + str(step) + '-' * 30)
		t_ = time.time()

		cur_block = generate_one_block(raw_graph, current_block_global_eid, srcnid, dstnid) # block -------

		t__=time.time()
		block_generation_time.append(t__-t_)  #------------------------------------------
		print('generate one block spend: ', t__-t_)
		#----------------------------------------------------
		induced_src = cur_block.srcdata[dgl.NID]
	
		e_src_local, e_dst_local = cur_block.edges(order='eid')
		e_src, e_dst = induced_src[e_src_local], induced_src[e_dst_local]
		e_src = e_src.detach().numpy().astype(int)
		e_dst = e_dst.detach().numpy().astype(int)

		#----------------------------------------------------
		blocks.append(cur_block)
		src_list.append(srcnid)
		dst_list.append(dstnid)

	connection_time = sum(check_connection_time)
	block_gen_time = sum(block_generation_time)

	return blocks, src_list, dst_list, (connection_time, block_gen_time)



def gen_batched_output_list(dst_nids, args ):
	batch_size=0
	if args.num_batch != 0 :
		batch_size = ceil(len(dst_nids)/args.num_batch)
		args.batch_size = batch_size
	print('number of batches is ', args.num_batch)
	print('batch size is ', batch_size)
 
	partition_method = args.selection_method
	batches_nid_list=[]
	weights_list=[]
	if partition_method=='range':
		indices = [i for i in range(len(dst_nids))]
		map_output_list = list(numpy.array(dst_nids)[indices])
		batches_nid_list = [map_output_list[i:i + batch_size] for i in range(0, len(map_output_list), batch_size)]
		length = len(dst_nids)
		weights_list = [len(batch_nids)/length  for batch_nids in batches_nid_list]
	if partition_method=='random':
		indices = torch.randperm(len(dst_nids))
		map_output_list = list(numpy.array(dst_nids)[indices])
		batches_nid_list = [map_output_list[i:i + batch_size] for i in range(0, len(map_output_list), batch_size)]
		length = len(dst_nids)
		weights_list = [len(batch_nids)/length  for batch_nids in batches_nid_list]

	return batches_nid_list, weights_list


def gen_grouped_dst_list(prev_layer_blocks):
	post_dst=[]
	for block in prev_layer_blocks:
		src_nids = block.srcdata['_ID'].tolist()
		post_dst.append(src_nids)
	return post_dst # return next layer's dst nids(equals prev layer src nids)


def generate_dataloader_wo_Betty_block(raw_graph, full_block_dataloader, args):
	data_loader=[]
	weights_list=[]
	num_batch=0
	blocks_list=[]
	final_dst_list =[]
	final_src_list=[]
	prev_layer_blocks=[]
	t_2_list=[]
	
	connect_checking_time_list=[]
	block_gen_time_total=0
	batch_blocks_gen_mean_time=0
	
	# the order of layer_block_list is bottom-up(the smallest block at first order)
	#b it means the graph partition starts 
	# from the output layer to the first layer input block graphs.
	for _,(src_full, dst_full, full_blocks) in enumerate(full_block_dataloader): # only one full batch blocks
		
		block_gen_time_total=0
		
		l=len(full_blocks)
		
		for layer_id, layer_block in enumerate(reversed(full_blocks)):
			# print('layer id ', layer_id)
			print('The real block id is ', l-1-layer_id)
		
			dst_nids=layer_block.dstdata['_ID']
			
			bb=time.time()
			block_eidx_global, block_edges_nids_global = get_global_graph_edges_ids_block(raw_graph, layer_block)
			get_eid_time=time.time()-bb
			print('get_global_graph_edges_ids_block function  spend '+ str(get_eid_time))
	

			layer_block.edata['_ID'] = block_eidx_global
			# eid_nids=layer_block.edata['_ID'].tolist() # global eids in this layer block
			if layer_id ==0:

				t1= time.time()
				batched_output_nid_list, weights_list=gen_batched_output_list(dst_nids, args)
				num_batch=len(batched_output_nid_list)
				
				select_time=time.time()-t1
				print(str(args.selection_method)+' selection method spend '+ str(select_time))
				# block 0 : (src_0, dst_0); block 1 : (src_1, dst_1);.......
				blocks, src_list, dst_list,time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block,  batched_output_nid_list)
				
				prev_layer_blocks=blocks
				blocks_list.append(blocks)
				final_dst_list=dst_list
				if layer_id==args.num_layers-1:
					final_src_list=src_list
			else:
				tmm=time.time()
				grouped_output_nid_list=gen_grouped_dst_list(prev_layer_blocks)
				print('gen group dst list time: ', time.time()-tmm)
				num_batch=len(grouped_output_nid_list)
				# print('num of batch ',num_batch )
				blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block, grouped_output_nid_list)

				if layer_id==args.num_layers-1: # if current block is the final block, the src list will be the final src
					final_src_list=src_list
				else:
					prev_layer_blocks=blocks
		
				blocks_list.append(blocks)
				connection_time, block_gen_time=time_1
				connect_checking_time_list.append(connection_time)
				block_gen_time_total+=block_gen_time
		# connect_checking_time_res=sum(connect_checking_time_list)
		batch_blocks_gen_mean_time= block_gen_time_total/num_batch
	
	
	for batch_id in range(num_batch):
		cur_blocks=[]
		for i in range(args.num_layers-1,-1,-1):
			cur_blocks.append(blocks_list[i][batch_id])
		
		dst = final_dst_list[batch_id]
		src = final_src_list[batch_id]
		data_loader.append((src, dst, cur_blocks))
	
	args.num_batch=num_batch
	
	






	return data_loader, weights_list, [sum(connect_checking_time_list), block_gen_time_total, batch_blocks_gen_mean_time]
	# return data_loader, weights_list, sum_list


def generate_dataloader_block(raw_graph, full_block_dataloader, args):
    
	if args.num_batch == 1:
		return full_block_dataloader,[1], [0, 0, 0]
	
	if 'REG' in args.selection_method or 'metis' in args.selection_method: # Betty
		return generate_dataloader_gp_block(raw_graph, full_block_dataloader, args)
	else: #'range' or 'random' in args.selection_method:
		return generate_dataloader_wo_Betty_block(raw_graph, full_block_dataloader, args)
	



def generate_dataloader_gp_block(raw_graph, full_block_dataloader, args):
	data_loader=[]
	weights_list=[]
	num_batch=0
	blocks_list=[]
	final_dst_list =[]
	final_src_list=[]
	prev_layer_blocks=[]
	t_2_list=[]
	
	connect_checking_time_list=[]
	block_gen_time_total=0
	batch_blocks_gen_mean_time=0
	
	full_blocks =[]

	# the order of layer_block_list is bottom-up(the smallest block at first order)
	#b it means the graph partition starts 
	# from the output layer to the first layer input block graphs.
	for _,(src_full, dst_full, full_blocks) in enumerate(full_block_dataloader): # only one full batch blocks
		
		block_gen_time_total=0
		l=len(full_blocks)
		for layer_id, layer_block in enumerate(reversed(full_blocks)):
			# print('layer id ', layer_id)
			
			print('The real block id is ', l-1-layer_id)
		
			# dst_nids=layer_block.dstdata['_ID']
			
			bb=time.time()
			block_eidx_global, block_edges_nids_global = get_global_graph_edges_ids_block(raw_graph, layer_block)
			get_eid_time=time.time()-bb
			print('get_global_graph_edges_ids_block function  spend '+ str(get_eid_time))
			layer_block.edata['_ID'] = block_eidx_global  # this only do in the first time

			# eid_nids=layer_block.edata['_ID'].tolist() # global eids in this layer block
			if layer_id ==0:

				t1= time.time()
				#----------------------------------------------------------
				my_graph_partitioner=Graph_Partitioner(layer_block, args) #init a graph partitioner object
				batched_output_nid_list,weights_list,batch_list_generation_time, p_len_list=my_graph_partitioner.init_graph_partition()

				print('partition_len_list')
				print(p_len_list)
				args.batch_size = my_graph_partitioner.batch_size
				#----------------------------------------------------------
				# batched_output_nid_list, weights_list=gen_batched_output_list(dst_nids, args.batch_size, args.selection_method)
				num_batch=len(batched_output_nid_list)
				
				#----------------------------------------------------------
				select_time=time.time()-t1
				print(str(args.selection_method)+' selection method  spend '+ str(select_time))
				# block 0 : (src_0, dst_0); block 1 : (src_1, dst_1);.......
				blocks, src_list, dst_list,time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block,  batched_output_nid_list)
				
				prev_layer_blocks=blocks
				blocks_list.append(blocks)
				final_dst_list=dst_list
				if layer_id==args.num_layers-1:
					final_src_list=src_list
			else:
				tmm=time.time()
				grouped_output_nid_list=gen_grouped_dst_list(prev_layer_blocks)
				print('gen group dst list time: ', time.time()-tmm)
				num_batch=len(grouped_output_nid_list)
				# print('num of batch ',num_batch )
				blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block, grouped_output_nid_list)

				if layer_id==args.num_layers-1: # if current block is the final block, the src list will be the final src
					final_src_list=src_list
				else:
					prev_layer_blocks=blocks
		
				blocks_list.append(blocks)
			connection_time, block_gen_time=time_1
			connect_checking_time_list.append(connection_time)
			block_gen_time_total+=block_gen_time
		# connect_checking_time_res=sum(connect_checking_time_list)
		batch_blocks_gen_mean_time= block_gen_time_total/num_batch

	for batch_id in range(num_batch):
		cur_blocks=[]
		for i in range(args.num_layers-1,-1,-1):
			cur_blocks.append(blocks_list[i][batch_id])
		
		dst = final_dst_list[batch_id]
		src = final_src_list[batch_id]
		data_loader.append((src, dst, cur_blocks))
	
	args.num_batch=num_batch

	##########################################################################################################
	check_t = 0
	b_gen_t = 0
	b_gen_t_mean = 0
	gp_time = 0
	for _ in range(args.num_re_partition):
		data_loader, weights_list, gp_time, [check_t,b_gen_t, b_gen_t_mean] = re_partition_block(raw_graph, full_blocks, args,  data_loader, weights_list)
	print('----------===============-------------===============-------------the number of batches *****----', len(data_loader))
	print()
	print('original number of batches: ',len(data_loader) - args.num_re_partition)
	args.num_batch = len(data_loader) - args.num_re_partition   #reset  the original number of batches
	if check_t:
		connect_checking_time_list = connect_checking_time_list + [check_t]

	block_gen_time_total = block_gen_time_total + b_gen_t
	batch_blocks_gen_mean_time = block_gen_time_total/len(data_loader)/args.num_layers
	print('re graph partition time: ', gp_time)
	print()
	##########################################################################################################

	return data_loader, weights_list, [sum(connect_checking_time_list), block_gen_time_total, batch_blocks_gen_mean_time]
	# return data_loader, weights_list, sum_list



def re_partition_block(raw_graph, full_blocks, args,  data_loader, weights_list):
	block_gen_time_total=0
	batch_blocks_gen_mean_time=0
	connect_checking_time_list=[]
	batch_list_generation_time_=0
	
	flag = False # change selection method
	b_id = intuitive_gp_first_layer_input_standard(args,  data_loader)

	if args.re_partition_method ==" " :
		return data_loader, weights_list, 0,[0,0,0]

	largest_batch=data_loader.pop(b_id) # pop the largest batch
	cur_blocks = list(largest_batch)[2]

	o_weight = weights_list.pop(b_id)
	

	new_num_batch = 2
	args.num_batch = new_num_batch
	# orignal_gp = args.selection_method
	############### change betty to random in re-partition*****
	if args.re_partition_method =='random' :
		flag = True
		args.selection_method = 'random' ############### change betty to random in re-partition**********
	############### change betty to random in re-partition*****
	### else: it will re-partition with Betty


	blocks_list=[]
	final_dst_list =[]
	final_src_list=[]
	
	for layer_id, layer_block in enumerate(reversed(full_blocks)):
	
			if layer_id == 0:		
				# we do re-partition start from  the smallest layer 
				my_graph_partitioner=Graph_Partitioner(cur_blocks[-1], args) #init a graph partitioner object
				batched_output_nid_list_, weights_list_, batch_list_generation_time_, p_len_list_ = my_graph_partitioner.init_graph_partition()
				weights_list_ = [ w/sum(weights_list_)*o_weight for w in weights_list_]
				# args.num_batch = original_num_batch + new_num_batch - 1

				blocks, src_list, dst_list,time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block,  batched_output_nid_list_)
						
				prev_layer_blocks=blocks
				blocks_list.append(blocks)
				
				final_dst_list =  dst_list
				if layer_id == args.num_layers-1:
					final_src_list =  src_list
					
			else:
				# tmm=time.time()
				grouped_output_nid_list=gen_grouped_dst_list(prev_layer_blocks)
				# print('gen group dst list time: ', time.time()-tmm)
				
				blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block, grouped_output_nid_list)

				if layer_id == args.num_layers-1: # if current block is the final block, the src list will be the final src
					final_src_list = src_list
				else:
					prev_layer_blocks = blocks
		
				blocks_list.append(blocks)
    
			connection_time, block_gen_time=time_1
			connect_checking_time_list.append(connection_time)
			block_gen_time_total += block_gen_time
	batch_blocks_gen_mean_time = block_gen_time_total/new_num_batch	

	for batch_id in range(new_num_batch):
		cur_blocks=[]
		for i in range(args.num_layers-1,-1,-1):
			cur_blocks.append(blocks_list[i][ batch_id])
		
		dst = final_dst_list[batch_id]
		src = final_src_list[batch_id]
		data_loader.append((src, dst, cur_blocks)) # add re-partition batches back to data_loader
		weights_list.append(weights_list_[batch_id])

	# args.num_batch = len(data_loader)

	if args.selection_method == 'random' and flag:
		args.selection_method = 'REG'

	return data_loader, weights_list, batch_list_generation_time_, [sum(connect_checking_time_list), block_gen_time_total, batch_blocks_gen_mean_time]





