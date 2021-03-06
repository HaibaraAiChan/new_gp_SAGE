#!/bin/bash


File=full_Betty_products_sage.py
Data=ogbn-products
Data=ogbn-arxiv
Data=cora
# Data=pubmed
# Data=reddit

model=sage
seed=1236 
setseed=True
GPUmem=True
load_full_batch=True
lr=0.01
dropout=0.5

run=1
epoch=1
logIndent=0

num_batch=(1)

pMethodList=(REG)

num_re_partition=(0)
# re_partition_method=REG
re_partition_method=random


layersList=(1)
# fan_out_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 15 20 25 30 40 50 100 150 200 400 800)
fan_out_list=(1 2 3 4 5 6 7 8 9 10 11 12 15 20 30 50 100 )
# fan_out_list=(1 2 3 4 5 10 50 100 200 300 400 500 800)
# fan_out_list=(2 3 4 )
hiddenList=(256)
AggreList=(pool)
aggre=pool
# AggreList=(mean)
# aggre=mean
savePath=./main_result/without_re/different_aggregator/${aggre}/1-epoch/${Data}/1-layer-REG/
# savePath=./main_result/without_re/different_aggregator/${aggre}/1-epoch/products/1-layer-REG/

for Aggre in ${AggreList[@]}
do      
	
	for pMethod in ${pMethodList[@]}
	do      
		
			for layers in ${layersList[@]}
			do      
				for hidden in ${hiddenList[@]}
				do
					for fan_out in ${fan_out_list[@]}
					do
						
						for nb in ${num_batch[@]}
						do
							
							for rep in ${num_re_partition[@]}
							do
								wf=${layers}-layer-fo-${fan_out}-sage-${Aggre}-h-${hidden}-batch-${nb}-gp-${pMethod}.log
								echo $wf

								python $File \
								--dataset $Data \
								--aggre $Aggre \
								--seed $seed \
								--setseed $setseed \
								--GPUmem $GPUmem \
								--selection-method $pMethod \
								--re-partition-method $re_partition_method \
								--num-re-partition $rep \
								--num-batch $nb \
								--lr $lr \
								--num-runs $run \
								--num-epochs $epoch \
								--num-layers $layers \
								--num-hidden $hidden \
								--dropout $dropout \
								--fan-out $fan_out \
								--log-indent $logIndent \
								--load-full-batch True \
								> ${savePath}${wf}

							done
						done
					done
				done
			done
		
	done
done
