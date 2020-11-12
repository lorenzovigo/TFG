#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_baselines.sh dataset_name


echo "Starting original experiments $1 ..."

python main.py --dataset $1 --algo_name mf --reindex > outputs_ML1M/original_$1_mf.txt
echo "DONE MF ORIGINAL"
python main.py --dataset $1 --algo_name fm --reindex > outputs_ML1M/original_$1_fm.txt
echo "DONE FM ORIGINAL"
python main.py --dataset $1 --algo_name nfm --reindex > outputs_ML1M/original_$1_nfm.txt
echo "DONE NFM ORIGINAL"

echo "Starting REINDEXED EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf > outputs_ML1M/reindexed_$1_mf.txt
echo "DONE MF REINDEXED"
python main.py --dataset $1 --algo_name fm > outputs_ML1M/reindexed_$1_fm.txt
echo "DONE FM REINDEXED"
python main.py --dataset $1 --algo_name nfm > outputs_ML1M/reindexed_$1_nfm.txt
echo "DONE NFM ORIGINAL"

echo "Starting GRAPH experiments..."
python main.py --dataset $1 --algo_name mf --gce > outputs_ML1M/graph_$1_mf.txt
echo "DONE MF REINDEXED-GCE"
python main.py --dataset $1 --algo_name fm --gce > outputs_ML1M/graph_$1_fm.txt
echo "DONE FM REINDEXED-GCE"
python main.py --dataset $1 --algo_name nfm --gce > outputs_ML1M/graph_$1_nfm.txt
echo "DONE NFM REINDEXED-GCE"


#echo "Starting GRAPH experiments ML MULTIHOP 2 ..."
#MF :      --prepro 10filter        --num_ng 8        --factors 92       --lr 0.0053
#FM :      --prepro 10filter        --num_ng 9        --factors 61       --lr 0.0031


#python main.py --dataset $1 --algo_name mf --reindex --prepro 10filter --num_ng 2 --factors 34 --lr 0.0005 > outputs/original/recommended_$1_mf.txt
#echo "DONE MF ORIGINAL"
#python main.py --dataset $1 --algo_name fm --reindex --prepro 10filter --num_ng 6 --factors 78 --lr 0.0015 > outputs/original/recommended_$1_fm.txt
#echo "DONE FM ORIGINAL"
#python main.py --dataset $1 --algo_name neumf --reindex > outputs/original/$1_neufm.txt
#echo "DONE NEUMF ORIGINAL"
#python main.py --dataset $1 --algo_name nfm --reindex > outputs/original/$1_nfm.txt
#echo "DONE NFM ORIGINAL"

#echo "Starting REINDEXED EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --prepro 10filter --num_ng 2 --factors 34 --lr 0.0005 > outputs/reindexed/recommended_$1_mf.txt
#echo "DONE MF REINDEXED"
#python main.py --dataset $1 --algo_name fm --prepro 10filter --num_ng 6 --factors 78 --lr 0.0015 > outputs/reindexed/recommended_$1_fm.txt
#echo "DONE FM REINDEXED"
#python main.py --dataset ml-100k --algo_name neumf > outputs/reindexed/neumf.txt
#echo "DONE NEUFM REINDEXED"
#python main.py --dataset $1 --algo_name nfm > outputs/reindexed/$1_nfm.txt
#echo "DONE NFM REINDEXED"

#echo "Starting GRAPH experiments..."
#python main.py --dataset $1 --algo_name mf --gce --prepro 10filter --num_ng 2 --factors 34 --lr 0.0005 > outputs/graph/recommended_$1_mf.txt
#echo "DONE MF REINDEXED-GCE"
#python main.py --dataset $1 --algo_name fm --gce --prepro 10filter --num_ng 6 --factors 78 --lr 0.0015 > outputs/graph/recommended_$1_fm.txt
#echo "DONE FM REINDEXED-GCE"
#python main.py --dataset ml-100k --algo_name neumf --gce > outputs/graph/$1_neumf.txt
#echo "DONE NEUFM REINDEXED"
#python main.py --dataset $1 --algo_name nfm  --gce > outputs/graph/$1_nfm.txt
#echo "DONE NFM REINDEXED-GCE"