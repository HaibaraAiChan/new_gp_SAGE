#!/bin/bash


# File=full_Betty_products_sage.py
File=mini_batch_train.py
Data=ogbn-products
# Data=ogbn-arxiv
model=sage
seed=1236 
setseed=True
GPUmem=True
load_full_batch=True
lr=0.01
dropout=0.5

run=1
epoch=5
logIndent=2

num_batch=(1 2 4 8 16 32 64)

layersList=(2)
fan_out_list=(10,25)

hiddenList=(256)
AggreList=(lstm)

# savePath=./main_result/without_re/different_aggregator/1-epoch/arxiv/1-layer-REG/
# savePath=./main_result/without_re/different_aggregator/1-epoch/products/1-layer-REG/
# savePath=./main_result/without_re/different_aggregator/1-epoch/products/2-layer-REG/
savePath=./dataloader_clean_folder/mini-batch-train/lstm/products/
for Aggre in ${AggreList[@]}
do      
	
			for layers in ${layersList[@]}
			do      
				for hidden in ${hiddenList[@]}
				do
					for fan_out in ${fan_out_list[@]}
					do
						
						for nb in ${num_batch[@]}
						do
							
							
								wf=${layers}-layer-fo-${fan_out}-sage-${Aggre}-h-${hidden}-batch-${nb}.log
								echo $wf

								python $File \
								--dataset $Data \
								--aggre $Aggre \
								--seed $seed \
								--setseed $setseed \
								--GPUmem $GPUmem \
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
