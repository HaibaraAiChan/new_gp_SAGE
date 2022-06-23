#!/bin/bash

File=pseudo_mini_batch_range_products_sage.py
gp=range
epoch=10
Aggre=mean
model=sage
seed=1236 
setseed=True
lr=0.01
dropout=0.5

layers=3
Data=ogbn-products

hidden=64
run=1
fan_out_list=(10,25,10 10,25,15 10,25,20 10,50,100 25,35,40 50,100,200)
batch_size=(98308)
nb=2
for fan_out in ${fan_out_list[@]}
do
        for bs in ${batch_size[@]}
        do
                python $File \
                --dataset $Data \
                --aggre $Aggre \
                --seed $seed \
                --setseed $setseed \
                --selection-method $gp \
                --batch-size $bs \
                --lr $lr \
                --num-runs $run \
                --num-epochs $epoch \
                --num-layers $layers \
                --num-hidden $hidden \
                --dropout $dropout \
                --fan-out $fan_out \
                --eval &> ../logs/sage/1_runs/${Data}_${Aggre}_${seed}_la_${layers}_fo_${fan_out}_nb_${nb}_run_${run}_ep_${epoch}.log
        done
done

batch_size=(49154)
nb=4
for fan_out in ${fan_out_list[@]}
do
        for bs in ${batch_size[@]}
        do
                python $File \
                --dataset $Data \
                --aggre $Aggre \
                --seed $seed \
                --setseed $setseed \
                --selection-method $gp \
                --batch-size $bs \
                --lr $lr \
                --num-runs $run \
                --num-epochs $epoch \
                --num-layers $layers \
                --num-hidden $hidden \
                --dropout $dropout \
                --fan-out $fan_out \
                --eval &> ../logs/sage/1_runs/${Data}_${Aggre}_${seed}_la_${layers}_fo_${fan_out}_nb_${nb}_run_${run}_ep_${epoch}.log
        done
done