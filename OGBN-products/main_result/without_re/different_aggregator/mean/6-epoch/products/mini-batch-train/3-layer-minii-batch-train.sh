#!/bin/bash


File=full_mini_batch_products_sage.py
Data=ogbn-products


model=sage
seed=1236 
setseed=True
GPUmem=True
load_full_batch=True
lr=0.01
dropout=0.5

run=1
epoch=6
logIndent=0

# num_batch=(1 2 4 8 16 32 64 128)
num_batch=(1 )

layersList=(3)
fan_out_list=(25,35,40)

hiddenList=(256)
AggreList=(mean)

savePath=./main_result/without_re/aggregator/6-epoch/products/mini-batch-train/


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
