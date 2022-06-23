import os
import numpy as np
import pandas as pd
from statistics import mean
import argparse
import copy
# import sys

# sys.path.insert(0,'..')
# sys.path.insert(0,'../..')




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
	res_dict.append(sorted(in_degree, reverse=True))
	print()
	print(len(res_dict))	
	print(len(res_dict[0]))	
	# print(res_dict)			
	
	for i in range(len(res_dict)):
		
		if (i+1)%3 ==0 and i<len(res_dict)-1:
			print(res_dict[i+1])
		if i% 48 == 0:
			print()	
	
	# for i in range(len(res_dict)):
	# 	# print(res_dict[i])
	# 	if (i+1)%3 ==0:
	# 		print()
			
def first_layer(filename, args, ):
	output=[]
	input=[]
	cnt = 0
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith('first layer output nodes number:'):
				output.append(int(line.split(' ')[-1]))
			if line.startswith('first layer input nodes number:'):
				
				input.append(int(line.split(' ')[-1]))
				
				
	# print(res_dict)			
	
	for i in range(len(input)):
		
		print(str(input[i]) + ', '+str(output[i]))
		
		if (i+1)% 16 == 0:
			print()	






if __name__=='__main__':
	
	print("computation info data collection start ...... " )
	argparser = argparse.ArgumentParser("info collection")
	# argparser.add_argument('--file', type=str, default='cora')
	# argparser.add_argument('--file', type=str, default='ogbn-products')
	argparser.add_argument('--file', type=str, default='ogbn-arxiv_lstm_l_3_fo_10,25,50_nb_16_random.log')
	# argparser.add_argument('--file', type=str, default='ogbn-arxiv_lstm_l_3_fo_10,25,50_nb_16_f7.log')
	
	# argparser.add_argument('--file', type=str, default='f7.log')
	# argparser.add_argument('--file', type=str, default='random.log')
	argparser.add_argument('--save-path',type=str, default='./')
	args = argparser.parse_args()
	file_in = args.file
	
	# in_degree(file_in,args,)	
	first_layer(file_in,args,)





