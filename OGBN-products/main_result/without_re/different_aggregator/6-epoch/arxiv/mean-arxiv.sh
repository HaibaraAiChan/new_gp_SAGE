#!/bin/bash


File=full_Betty_products_sage.py
# Data=ogbn-arxiv
Data=ogbn-products
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

num_batch=(1 )
# num_batch=( 2 4 8 10 16 32 64)
# num_batch=(32 )
# num_batch=(16 ) 

pMethodList=(REG)
# pMethodList=(random range)
# pMethodList=(metis)

num_re_partition=(0)
# re_partition_method=REG
re_partition_method=random


# layersList=(2)
# fan_out_list=(10,25)
layersList=(1)
fan_out_list=(10)
hiddenList=(64)
AggreList=(mean)

savePath=./main_result/without_re/different_aggregator/6-epoch/arxiv/mean-betty/products_1_epoch


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
								wf=${layers}-layer-fo-${fan_out}-sage-${Aggre}-h-${hidden}-batch-${nb}-rep-${rep}.log
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
