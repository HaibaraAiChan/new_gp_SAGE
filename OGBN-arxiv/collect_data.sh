#!/bin/bash

# folderPath=/home/cc/graph_partition_multi_layers/pseudo_mini_batch_full_batch/SAGE

File=data_collection.py
Data=ogbn-arxiv

savePath=./logs
# mkdir ${savePath}/
# logPath=${savePath}/

model=sage


# pMethodList=(shared_neighbor_graph_partition random range shared_neighbor_graph_partition_set_one) 
pMethodList=(shared_neighbor_graph_partition_s0) 

layersList=(1)


hiddenList=(256 )
AggreList=(lstm )


for Aggre in ${AggreList[@]}
do      
        for pMethod in ${pMethodList[@]}
        do      
                for layers in ${layersList[@]}
                do      
                        for hidden in ${hiddenList[@]}
                        do
                                echo 'text'
                                python $File \
                                        --file $Data \
                                        --aggre $Aggre \
                                        --model $model \
                                        --selection-method $pMethod \
                                        --num-layers $layers \
                                        --hidden $hidden \
                                        &> ./logs/${pMethod}/${layers}_layer_${Aggre}__h_${hidden}_eff.log
                                        
                                        
                        done
                done
        done
done
