import os
import numpy as np
import pandas as pd
from statistics import mean
import argparse
import sys



def calculate_redundancy(dict_nods):
	dict_nods = dict(sorted(dict_nods.items()))
	nb = list(dict_nods.keys())
	eff = list(dict_nods.values())
	res={}
	if nb[1]:
		base = eff[0]
		for i in range(1,len(eff)):
			red = eff[i]/base
			# print(str(nb[i]) +' batches redundancy: '+str(red))
			res[nb[i]] = red
	print()
	print(res)
	print()
	

def to_df(filename, loss_list, test_acc_list):
	dl = pd.DataFrame({'loss': loss_list, })
	dac = pd.DataFrame({ 'test_acc':test_acc_list})
	# print(dd[:10])
	dl.to_csv(filename + '_loss_.dat',  sep='\t')
	dac.to_csv(filename + '_acc_.dat',  sep='\t')

def get_num_batch(filename):
	num_batch =filename.split('-')[11]
	return num_batch

def first_layer_output_nodes(infile):
	f = open(infile,'r')
	nid = []
	for line in f:
		line = line.strip()
		if line.startswith("Number of first layer output nodes during this epoch:"):
			nid.append(int(line.split(' ')[-1]))
	f.close()
	return mean(nid)

def feature_label_block_edges_movement_time(infile):
	f = open(infile,'r')
	feat_label_time = []
	block_edges_time = []
	for line in f:
		line = line.strip()
		if line.startswith("load block tensor time/epoch"):
			feat_label_time.append(float(line.split(' ')[-1]))
		if line.startswith("block to device time/epoch"):
			block_edges_time.append(float(line.split(' ')[-1]))
	f.close()
	f_t = mean(feat_label_time[2:])
	e_t = mean(block_edges_time[2:])
	return f_t+e_t


def first_layer_input_nodes(infile):
	f = open(infile,'r')
	first_input_nid = []
	
	for line in f:
		line = line.strip()
		
		if line.startswith("Number of first layer input nodes during this epoch:"):
			first_input_nid.append(int(line.split(' ')[-1]))
	f.close()
	return mean(first_input_nid)

def computation_nodes(infile):
	f = open(infile,'r')
	
	compute_num_nid = []
	
	for line in f:
		line = line.strip()
		if line.startswith("Number of nodes for computation during this epoch:"):
			compute_num_nid.append(int(line.split(' ')[-1]))
	f.close()
	return mean(compute_num_nid)

def computation_eff(infile):
	f = open(infile,'r')
	
	compute_num_nid = []
	first_input_nid = []
	pure_train_times = []
	for line in f:
		line = line.strip()
		if line.startswith("Number of nodes for computation during this epoch:"):
			compute_num_nid.append(int(line.split(' ')[-1]))
			# print(int(line.split(' ')[-1]))
		if line.startswith("Number of first layer input nodes during this epoch:"):
			first_input_nid.append(int(line.split(' ')[-1]))
		if line.startswith("Training time without total dataloading part /epoch "):
			pure_train_times.append(float(line.split(' ')[-1]))	
	
	if sum(pure_train_times)==0:
			return 0,0,0
	if len(pure_train_times)>1: 
		pure_train_times = pure_train_times[3:-1]
	real_pure_train_eff = mean(compute_num_nid)/mean(pure_train_times)
	# print('mean computation nodes: ',mean(compute_num_nid))
	pure_train_t = mean(pure_train_times)

	input_eff = mean(first_input_nid)/mean(pure_train_times)
	# print('mean first layer input nodes: ',mean(first_input_nid))
	
	f.close()
	return input_eff, real_pure_train_eff, pure_train_t

def Memory_usage(infile):
	f = open(infile,'r')
	max_mem = []
	
	for line in f:
		line = line.strip()
		if line.startswith("Max Memory Allocated:"):
			if float(line.split(' ')[3]):
				max_mem.append(float(line.split(' ')[3]))

	return max_mem[-1]
		
if __name__=='__main__':
	res={}
	in_ = {}
	out_ = {}
	in_eff_res={}
	train_time={}
	data_move_time={}
	mem={}
	computation_nid = {}
	for filename in os.listdir("./"):
		if filename.endswith(".log") : # and filename.startswith('3')
			num_batch = int(get_num_batch(filename))
			print(num_batch)
			# in_eff, eff, t = computation_eff(filename)
			# in_eff_res[num_batch] =  in_eff
			# res[num_batch]=eff
			# train_time[num_batch]=t
			mem_tmp = Memory_usage(filename)
			mem[num_batch]=mem_tmp

			# in_[num_batch] = first_layer_input_nodes(filename)
			# out_[num_batch] = first_layer_output_nodes(filename)

			# computation_nid[num_batch] = computation_nodes(filename)
			# data_move_time[num_batch] = feature_label_block_edges_movement_time(filename)

	# print('computation input nodes number')
	# print(dict(sorted(in_.items())))
	# print('computation input nodes redundancy')
	# calculate_redundancy(in_)
	# print('-'*80)

	# print('computation ***** first layer output nodes number ')
	print()
	# print(dict(sorted(out_.items())))
	# print('computation ***** first layer output nodes redundancy')
	# calculate_redundancy(out_)
	# print('='*80)
	# print('computation input eff---------------------------throughput-------------')
	# print(dict(sorted(in_eff_res.items())))

	# print('computation nodes')
	# print(dict(sorted(computation_nid.items())))
	# print('='*80)
	# print()
	# print('computation eff')
	# print(dict(sorted(res.items())))
	# print('='*80)
	# print()
	# print('time')
	# print(dict(sorted(train_time.items())))
	print('cuda max mem')
	print(dict(sorted(mem.items())))
	# print()
	# print('='*80)
	# print('computation ***** first layer input nodes redundancy')
	# calculate_redundancy(in_)
	# print('='*80)

	# print('='*80)
	# print('data movement time')
	# print(dict(sorted(data_move_time.items())))
	# print('='*80)
