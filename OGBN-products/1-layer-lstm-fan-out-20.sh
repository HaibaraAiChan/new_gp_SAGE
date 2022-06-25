#!/bin/bash


File=full_Betty_products_sage.py
Data=ogbn-products

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

num_batch=(2)
num_batch=(3 4 5 6 7 8 16 )
pMethodList=(REG)

num_re_partition=(0)
# re_partition_method=REG
re_partition_method=random


layersList=(1)
fan_out_list=(20)

hiddenList=(256)
AggreList=(lstm)


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
							> ./main_result/without_re/fan-out/${layers}-layer-fo-${fan_out}-sage-${Aggre}-batch-${nb}-rep-${rep}.log

							done
						done
					done
				done
			done
		
	done
done