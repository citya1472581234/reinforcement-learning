#!/usr/bin/env bash

# Problem 4
#python train_pg_f18.py CartPole-v0 -n=100 -b=1000 -e=3 -dna --exp_name sb_no_rtg_dna
#wait
#python train_pg_f18.py CartPole-v0 -n=100 -b=1000 -e=3 -rtg -dna --exp_name sb_rtg_dna
#wait
#python train_pg_f18.py CartPole-v0 -n=100 -b=1000 -e=3 -rtg --exp_name sb_rtg_na
#wait
#python train_pg_f18.py CartPole-v0 -n=100 -b=5000 -e=3 -dna --exp_name lb_no_rtg_dna
#wait
#python train_pg_f18.py CartPole-v0 -n=100 -b=5000 -e=3 -rtg -dna --exp_name lb_rtg_dna
#wait
#python train_pg_f18.py CartPole-v0 -n=100 -b=5000 -e=3 -rtg --exp_name lb_rtg_na

# Problem 5
#for lr in 0.00001 0.0001 0.001 0.01 0.1 1
#do
#    for b in 10 100 1000 5000
#    do
#        python train_pg_f18.py InvertedPendulum-v2 -ep=1000 --discount=0.9 \
#         -n=100 -e=3 -l=2 -s=64 -b=$b -lr=$lr -rtg --exp_name=ip_b$b\_r$lr
#    done
#done

# Problem 7
# Limited by the GPU memory, I have to reduce batch size from 40000 to 10000,
# and reduce the number of process from 3 to 2.
nohup bash one_exp.sh &
