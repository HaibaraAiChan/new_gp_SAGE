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


if __name__=='__main__':
	# filename = 'products-full-3-layer-fo-10,25,50-mean-200-epoch-h-64-lr-0.001.log'
	filename = 'full-500.log'
	# filename = 'products-16-batch-3-layer-fo-10,25,50-mean-500-epoch-h-64-lr-0.001.log'
	compute_efficiency_full_graph(filename)	
	filename = '16-batch.log'
	compute_efficiency_full_graph(filename)	
