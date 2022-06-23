import os
import numpy as np
import pandas as pd
from statistics import mean
import argparse
import sys



def compute_efficiency_full_graph(filename):
	
	loss_list = []
	test_acc_list = []
	with open(filename) as f:
		train_times = []
		for line in f:
			line = line.strip()
			if line.startswith("Run 00 ") :
				ll = line.split(' ')
				loss = float(ll[7])
				test_acc = float(ll[-1])
				print(str(loss)+', '+str(test_acc))
				test_acc_list.append(test_acc)
				loss_list.append(loss)
	print()
	print('test acc ')
	print(test_acc_list)
	print()
	print('loss list')
	print(loss_list)
	to_df(filename, loss_list, test_acc_list)

	

def to_df(filename, loss_list, test_acc_list):
	dl = pd.DataFrame({'loss': loss_list, })
	dac = pd.DataFrame({ 'test_acc':test_acc_list})
	# print(dd[:10])
	dl.to_csv(filename + '_loss_.dat',  sep='\t')
	dac.to_csv(filename + '_acc_.dat',  sep='\t')

def get_num_batch(filename):
	num_batch =filename.split('-')[2]
	return num_batch

def computation_eff(infile):
	f = open(infile,'r')
	compute_num_nid = []
	pure_train_times = []
	for line in f:
		line = line.strip()
		if line.startswith("Number of nodes for computation during this epoch:"):
			compute_num_nid.append(int(line.split(' ')[-1]))
		if line.startswith("Training time without total dataloading part /epoch "):
			pure_train_times.append(float(line.split(' ')[-1]))	
		
	real_pure_train_eff = sum(compute_num_nid)/sum(pure_train_times)
	return real_pure_train_eff
		

if __name__=='__main__':
	res={}
	for filename in os.listdir("./"):
		if filename.endswith(".log"):
			num_batch = int(get_num_batch(filename))
			eff = computation_eff(filename)
			res[num_batch]=eff
	print(dict(sorted(res.items())))
	# df=pd.DataFrame(res).transpose()
	# df_res.index.name
	# # df.columns=['num of batch','total nodes for computation/pure train time']
	# print(df.to_markdown(tablefmt="grid"))

			
	
