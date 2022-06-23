#!/bin/bash


File=full_and_pseudo_mini_batch_arxiv_sage.py
Data=ogbn-arxiv



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

# pMethodList=(shared_neighbor_graph_partition random range shared_neighbor_graph_partition_set_one) 
pMethodList=(shared_neighbor_graph_partition_s0)
# pMethodSubList=(f0 f1 f2 f3 f4 f5 f6 f7) 
pMethodSubList=(f3 ) 
# re_partition_method=Betty

re_partition_method=random # when set --num-re-partition to 0, it doesn't matter the re gp method
# pMethodList=( random range) 
# pMethodList=( random ) 
# pMethodSubList=(f0) 

# num_batch=( 2 4 8 16 32 64 )

num_batch=(1 128 256)
# num_batch=( 2 )
# num_re_partition=(1 2 3 4 5 6)
# num_batch=(2 )
layersList=(1)
fan_out_list=(10)
hiddenList=(256 )
AggreList=(lstm )

savePath=logs/1-layer

# mkdir ${savePath}
for Aggre in ${AggreList[@]}
do      
        # mkdir ${savePath}/${Aggre}/
        for pMethod in ${pMethodList[@]}
        do      
                for pMethodSub in ${pMethodSubList[@]}
                do 
                        # mkdir ${savePath}/${Aggre}/drop_edges
                        for layers in ${layersList[@]}
                        do      
                                for hidden in ${hiddenList[@]}
                                do
                                        for fan_out in ${fan_out_list[@]}
                                        do
                                                rep=0
                                                for nb in ${num_batch[@]}
                                                do
                                                        
                                                
                                                        python $File \
                                                        --dataset $Data \
                                                        --aggre $Aggre \
                                                        --seed $seed \
                                                        --setseed $setseed \
                                                        --GPUmem $GPUmem \
                                                        --selection-method $pMethod \
                                                        --selection-method-sub  $pMethodSub \
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
                                                        &> ${savePath}/${Aggre}/no-re-gp/${Data}_${Aggre}_l_${layers}_fo_${fan_out}_nb_${nb}.log

                                                        # rep=$(($rep+1))
                                                done
                                        done
                                done
                        done
                done
        done
done

# pMethodList=( random range) 
# pMethodSubList=(f) 
# for Aggre in ${AggreList[@]}
# do      
#         # mkdir ${savePath}/${Aggre}/
#         for pMethod in ${pMethodList[@]}
#         do      
#                 for pMethodSub in ${pMethodSubList[@]}
#                 do 
#                         # mkdir ${savePath}/${Aggre}/drop_edges
#                         for layers in ${layersList[@]}
#                         do      
#                                 for hidden in ${hiddenList[@]}
#                                 do
#                                         for fan_out in ${fan_out_list[@]}
#                                         do
#                                                 for nb in ${num_batch[@]}
#                                                 do
                                                        
                                                
#                                                         python $File \
#                                                         --dataset $Data \
#                                                         --aggre $Aggre \
#                                                         --seed $seed \
#                                                         --setseed $setseed \
#                                                         --GPUmem $GPUmem \
#                                                         --selection-method $pMethod \
#                                                         --num-batch $nb \
#                                                         --lr $lr \
#                                                         --num-runs $run \
#                                                         --num-epochs $epoch \
#                                                         --num-layers $layers \
#                                                         --num-hidden $hidden \
#                                                         --dropout $dropout \
#                                                         --fan-out $fan_out \
#                                                         --log-indent $logIndent \
#                                                         --load-full-batch True \
#                                                         &> ${savePath}/${Aggre}/drop_edges/${Data}_${Aggre}_l_${layers}_fo_${fan_out}_nb_${nb}_${pMethod}.log

#                                                         # nb=$(($nb*2))
#                                                 done
#                                         done
#                                 done
#                         done
#                 done
#         done
# done
