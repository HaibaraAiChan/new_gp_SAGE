import os
import numpy as np
import pandas as pd
from statistics import mean
import argparse
import copy
# import sys

# sys.path.insert(0,'..')
# sys.path.insert(0,'../..')

def get_fan_out(filename):
	fan_out=filename.split('_')[5]
	# print(fan_out)
	return fan_out
def get_num_batch(filename):
	nb=filename.split('_')[7][:-4]
	# print(nb)
	return nb


def in_degree(filename, args, ):
	res_dict=[]
	in_degree=[]
	cnt = 0
	with open(filename) as f:
		for line in f:
			line = line.strip()
			
			if line.startswith('m.shape torch.Size'):
				
				res = line.split('[')[-1].split(',')
				
				if int(res[1]) == 1:
					if cnt >= 1:
						print()
						print(in_degree)
						res_dict.append(sorted(in_degree, reverse=True))
						in_degree = []
						print()
						print('-='*50)
					cnt +=1
				in_degree.append(float(res[0]))
				print(line)
	print()
	print(in_degree)
	res_dict.append(sorted(in_degree, reverse=True))
	print()
	print(len(res_dict))	
	print(len(res_dict[0]))	
	# print(res_dict)			
	
	for i in range(len(res_dict)):
		print(res_dict[i])
		if (i+1)% 2 == 0:
			print()	
	
	# for i in range(len(res_dict)):
	# 	# print(res_dict[i])
	# 	if (i+1)%3 ==0:
	# 		print()
			
def first_layer(filename, args, ):
	output=[]
	input_=[]
	cnt = 0
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith('output nodes number:'): # 1-layer model is special
				output.append(int(line.split(' ')[-1]))
			if line.startswith('input nodes number:'):
				input_.append(int(line.split(' ')[-1]))
				
				
	# print(res_dict)			
	
	for i in range(len(input_)):
		
		print(str(input_[i]) + ', '+str(output[i]))
		
		if (i+1)% 2 == 0:
			print()	
	return sum(input_)





def pure_train_time( filename, args, nb):
	res={}
	pure_train_time=[]
	train_wo_tensor_to_gpu =[]
	computation_nodes=[]
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith('* Pure training time/epoch'): # 1-layer model is special
				pure_train_time.append(float(line.split(' ')[-1]))
			if line.startswith('Training time without total dataloading part /epoch'):
				train_wo_tensor_to_gpu.append(float(line.split(' ')[-1]))
			if line.startswith("Number of nodes for computation during this epoch: "):
				computation_nodes.append(int(line.split(' ')[-1]))
	# print('train time (tensor & blocks to gpu): ', mean(pure_train_time))
	# print('train time (wo tensor & blocks to gpu): ',  mean(train_wo_tensor_to_gpu))
	res.update({'train time (tensor & blocks to gpu)': mean(pure_train_time)})
	res.update({'train time (wo tensor & blocks to gpu)': mean(train_wo_tensor_to_gpu)})
	res.update({'train time (wo tensor & blocks to gpu)/batch': mean(train_wo_tensor_to_gpu)/nb})
	res.update({'# of computation nodes(wo tensor & blocks to gpu)/batch': mean(computation_nodes)/nb})
	return res

def get_first_layer_output_size(filename):
	first_layer_num_output_nid=[]
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith('Number of first layer output nodes during this epoch:'):
				first_layer_num_output_nid.append(int(line.split(' ')[-1])) 
			elif line.startswith('first layer output nodes number:'): # when OOM, the first layer output of full batch 
				first_layer_num_output_nid.append(int(line.split(' ')[-1])) 

	if len(first_layer_num_output_nid)==0:
		return 0
	return int(mean(first_layer_num_output_nid))




def get_full_batch_input_size(filename):
	first_layer_num_input_nid=[]
	
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith('Number of first layer input nodes during this epoch:'):
				first_layer_num_input_nid.append(int(line.split(' ')[-1])) 
			elif line.startswith('first layer input nodes number:'): # when OOM, the first layer output of full batch 
				first_layer_num_input_nid.append(int(line.split(' ')[-1])) 
			
	if len(first_layer_num_input_nid)==0:
		return 0
	return int(mean(first_layer_num_input_nid))

def train_time(args):
	res=[]
	column_names=[]
	path =args.path
	nb_list=[]
	fan_out =""
	for f_item in os.listdir(args.path):
		if 'nb_' in f_item:
			nb_size=f_item.split('_')[7][:-4]
			nb_list.append(int(nb_size))
	nb_list.sort()
	nb_list=['ogbn-arxiv_lstm_l_1_fo_10_nb_'+str(i)+".log" for i in nb_list]
	for f_item in nb_list:
		# path_r=path + f_item
	
		if f_item.endswith(".log"):
			f = os.path.join(path, f_item)
			fan_out=str(get_fan_out(f_item))
			nb=get_num_batch(f_item)
			if int(nb) == 1:
				column_names.append('full batch \n'+fan_out)
				# column_names_csv.append('full batch '+fan_out)
				# full_batch_input_size=get_full_batch_input_size(f)
				# full_batch_output_size_first_layer=get_first_layer_output_size(f)
				# dict2 = pure_train_time(f, args,)
			else:
				column_names.append('pseudo \n'+str(nb)+' batches\n'+fan_out)
				# column_names_csv.append('pseudo '+str(nb)+' batches'+fan_out)
			dict2 = pure_train_time(f,  args, int(nb))
			res += [dict2]
	df=pd.DataFrame(res).transpose()
	df.columns=column_names
	df.index.name=args.data+' '+args.model +' fan-out '+fan_out+' hidden '+str(args.hidden)
	print(df.to_markdown(tablefmt="grid"))









if __name__=='__main__':
	
	print("computation info data collection start ...... " )
	argparser = argparse.ArgumentParser("info collection")
	# argparser.add_argument('--file', type=str, default='cora')
	# argparser.add_argument('--file', type=str, default='ogbn-products')
	# argparser.add_argument('--file', type=str, default='ogbn-arxiv_lstm_l_1_fo_10_nb_2_random.log')
	argparser.add_argument('--file', type=str, default='./lstm/no-re-gp/ogbn-arxiv_lstm_l_1_fo_10_nb_2.log')
	argparser.add_argument('--model', type=str, default='sage')
	argparser.add_argument('--data', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--file', type=str, default='f7.log')
	# argparser.add_argument('--file', type=str, default='random.log')
	argparser.add_argument('--save-path',type=str, default='./')
	argparser.add_argument('--path',type=str, default='./lstm/no-re-gp/')
	argparser.add_argument('--hidden', type=int, default=256)
	args = argparser.parse_args()
	file_in = args.file
	
	# in_degree(file_in,args,)	
	first_layer(file_in,args,)
	# pure_train_time(file_in,args,)
	train_time(args)





