#!/bin/bash

# folderPath=/home/cc/graph_partition_multi_layers/pseudo_mini_batch_full_batch/SAGE

File=full_Betty_products_sage.py
Data=ogbn-products

savePath=../../logs/sage/1_runs/pure_train/ogbn_products

model=sage
seed=1236 
setseed=True
GPUmem=True
load_full_batch=True
lr=0.01
dropout=0.5


run=1
epoch=3
logIndent=1

pMethodList=(random range) 

num_batch=(1 2 4 8 16 32 64 128 256)

layersList=(3)
fan_out_list=(25,35,40)

hiddenList=(256 )
AggreList=(mean)

mkdir ${savePath}
for Aggre in ${AggreList[@]}
do      
        mkdir ${savePath}/${Aggre}/
        for pMethod in ${pMethodList[@]}
        do      
                mkdir ${savePath}/${Aggre}/${pMethod}
                for layers in ${layersList[@]}
                do      
                        mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/
                        for hidden in ${hiddenList[@]}
                        do
                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/
                                for fan_out in ${fan_out_list[@]}
                                do
                                        # nb=1
                                        for nb in ${num_batch[@]}
                                        do
                                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                logPath=${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                # mkdir $logPath
                                                echo $logPath
                                                echo 'number of batches'
                                                echo $nb
                                                python $File \
                                                --dataset $Data \
                                                --aggre $Aggre \
                                                --seed $seed \
                                                --setseed $setseed \
                                                --GPUmem $GPUmem \
                                                --selection-method $pMethod \
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
                                                &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log

                                                # nb=$(($nb*2))
                                        done
                                done
                        done
                done
        done
done
