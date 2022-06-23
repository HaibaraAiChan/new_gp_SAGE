#!/bin/bash


File=full_and_pseudo_mini_batch_products_sage.py
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

# pMethodList=(shared_neighbor_graph_partition random range shared_neighbor_graph_partition_set_one) 
pMethodList=(shared_neighbor_graph_partition_s0)
# pMethodSubList=(f0 f1 f2 f3 f4 f5 f6 f7) 
pMethodSubList=(f3 ) 
# re_partition_method=' '
# re_partition_method=shared_neighbor_graph_partition_s0
# re_partition_method=random
# pMethodList=( random range) 
# pMethodList=( random ) 
# pMethodSubList=(f0) 
num_batch=( 8 )
# num_batch=( 4 8 16 32)
# num_batch=( 64 128 256)
# num_batch=( 512 1024 2048)
# num_re_partition=(1 2 3 4 5 6)

layersList=(4)
fan_out_list=(10,25,30,40)
hiddenList=(256 )
AggreList=( mean )



for Aggre in ${AggreList[@]}
do      
        for pMethod in ${pMethodList[@]}
        do      
                for pMethodSub in ${pMethodSubList[@]}
                do 
                        for layers in ${layersList[@]}
                        do      
                                for hidden in ${hiddenList[@]}
                                do
                                        for fan_out in ${fan_out_list[@]}
                                        do
                                                rep=1  ##### $ number of re-partition
                                                for nb in ${num_batch[@]}
                                                do
                                                        
                                                
                                                        python $File \
                                                        --dataset $Data \
                                                        --aggre $Aggre \
                                                        --seed $seed \
                                                        --setseed $setseed \
                                                        --GPUmem $GPUmem \
                                                        --selection-method $pMethod \
                                                        --selection-method-sub $pMethodSub \
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
                                                        > ./main_result/without_re/num_of_layers/4-layer-sage-${Aggre}-batch-${nb}.log

                                                        # rep=$(($rep+1))
                                                done
                                        done
                                done
                        done
                done
        done
done
