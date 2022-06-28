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
	
	if sum(pure_train_times)==0:
			return 0
	if len(pure_train_times)>1: 
		pure_train_times = pure_train_times[1:]
	real_pure_train_eff = mean(compute_num_nid)/mean(pure_train_times)
	print('mean nodes: ',mean(compute_num_nid))
	pure_train_t = mean(pure_train_times)
	
	return real_pure_train_eff, pure_train_t

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
	train_time={}
	mem={}
	for filename in os.listdir("./"):
		if filename.endswith("6.log"):
			num_batch = int(get_num_batch(filename))
			eff, t = computation_eff(filename)
			res[num_batch]=eff
			train_time[num_batch]=t
			# mem_tmp = Memory_usage(filename)
			# mem[num_batch]=mem_tmp
	print('computation eff')
	print(dict(sorted(res.items())))
	print('time')
	print(dict(sorted(train_time.items())))

# if __name__=='__main__':
# 	res={}
# 	train_time={}
# 	mem={}
# 	for filename in os.listdir("./"):
# 		if filename.endswith(".log"):
# 			num_batch = int(get_num_batch(filename))
# 			eff, t = computation_eff(filename)
# 			res[num_batch]=eff
# 			train_time[num_batch]=t
# 			mem_tmp = Memory_usage(filename)
# 			mem[num_batch]=mem_tmp
# 	print('computation eff')
# 	print(dict(sorted(res.items())))
# 	print('time')
# 	print(dict(sorted(train_time.items())))
# 	print()
# 	print('cuda max mem')
# 	print(dict(sorted(mem.items())))
# 	print()
# 	# df=pd.DataFrame(res).transpose()
# 	# df_res.index.name
# 	# # df.columns=['num of batch','total nodes for computation/pure train time']
# 	# print(df.to_markdown(tablefmt="grid"))

			
# 	# data = {3: 7233092.558344982, 4: 7526558.609479932, 5: 10281525.910010189, 6: 9637163.811590359, 7: 15148023.417957973, 8: 14582559.197133293, 16: 18014837.01344873, 32: 23984937.36776107}
# # data = dict(sorted(data.items()))
# # x=list(data.keys())
# # y=list(data.values())
# # plt.plot(x, y)
# # plt.bar(x, y)

# t={3: 2.4283907413482666, 4: 2.913294792175293, 5: 2.5525121688842773, 6: 3.1250216960906982, 7: 2.218050241470337, 8: 2.5884571075439453, 16: 3.4711878299713135, 32: 4.516044855117798}
# x=list(t.keys())
# y2=list(t.values())
# plt.plot(x, y2)

# plt.show()
