import os
import numpy as np
import pandas as pd
from statistics import mean
import argparse
import sys



def ss(line):
	data_ = line.split(',')[1:]
	print('data_')
	# print(data_)
	rest=[]
	for item in data_:
		rest.append(float(item.split(':')[1]))
	
	print(rest)
	return rest


def cal(infile):
	f = open(infile,'r')
	random = []
	betty =[]
	for line in f:
		line = line.strip()
		if line.startswith("Random"):
			random = ss(line)
			
		if line.startswith("Betty"):
			betty = ss(line)

	diff = [i - j for i, j in zip(random, betty)]
	res = [i / j for i, j in zip(diff, random)]
	print('-'*50)
	print('res')
	print(res)

	return 
		
if __name__=='__main__':
	# random =[1.3800601959228516, 1.2265548706054688, 1.0193052291870117,  0.8023581504821777, 0.642885684967041,  0.5169777870178223,  0.3821725845336914]
	# betty = [1.3800601959228516,  0.978818416595459, 0.7145800590515137, 0.583594799041748, 0.44493532180786133, 0.35678815841674805, 0.30204248428344727]
	
	# diff = [i - j for i, j in zip(random, betty)]
	# res = [i / j for i, j in zip(diff, random)]

	# print(res)
	
	
	for filename in os.listdir("./"):
		if filename.endswith(".txt") :
			cal(filename)