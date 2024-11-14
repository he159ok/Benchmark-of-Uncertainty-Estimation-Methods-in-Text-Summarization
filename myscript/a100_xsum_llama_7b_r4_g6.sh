#!/bin/bash

#SBATCH --account=stgnn
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -J f_xr4_g6_llama
#SBATCH -o ./log/f_xr4_g6_llama.output.txt
#SBATCH -e ./log/f_xr4_g6_llama.error.txt

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
    meta-llama/Llama-2-7b-chat-hf \
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
    -use_chatgpt_35_def \
    1 \
    -use_chatgpt_35_def_0 \
    1 \
    -use_chatgpt_35_def_both \
    1 \
    -use_chatgpt_40_def \
    0 \
    -use_claude \
    0 \
    -use_unieval_overall \
    1 \
    -ue_cal_name \
    white_six_full_r4_g6 \
    -use_ensemble_ue \
    0 \
    -use_part \
    g6 \

