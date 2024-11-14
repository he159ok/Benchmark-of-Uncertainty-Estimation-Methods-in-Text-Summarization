#!/bin/bash

#SBATCH --account=stgnn
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -J f_xbt_g1
#SBATCH -o ./log/f_xbt_g1.output.txt
#SBATCH -e ./log/f_xbt_g1.error.txt

#module spider anaconda
#module load Anaconda3/2020.11
#source activate ~/env/unc_kg



dname=$(dirname "$PWD")
cd $dname
echo $dname

CUDA_VISIBLE_DEVICES=0,

python main_ats_revise.py \
    -model_type \
    w \
    -api_token \
    XXX \
    -model_name_or_path \
    facebook/bart-large-cnn \
    -device \
    cuda:0 \
    -dataset_name \
    xsum \
    -batch_size \
    1 \
    -seed \
    42 \
    -use_small \
    0 \
    -use_rouge \
    1 \
    -use_bart \
    1 \
    -use_summac \
    1 \
    -use_ctc \
    1 \
    -use_spearmanr \
    1 \
    -use_kendalltau \
    1 \
    -use_chatgpt \
    1 \
    -use_claude \
    0 \
    -use_unieval_overall \
    1 \
    -ue_cal_name \
    white_six_full_bart_g1 \
    -use_ensemble_ue \
    1 \
    -use_part \
    g1 \

