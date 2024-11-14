#!/bin/bash

#SBATCH --account=stgnn
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH -J f_tof_gpt_rhum3
#SBATCH -o ./log/f_tof_gpt_rhum3_def35both.output.txt
#SBATCH -e ./log/f_tof_gpt_rhum3_def35both.error.txt

#module spider anaconda
#module load Anaconda3/2020.11
#source activate /home/jianfenghe/env/unc_kg
#
#which python




dname=$(dirname "$PWD")
cd $dname
echo $dname

CUDA_VISIBLE_DEVICES=0,

python main_ats_revise_tofu.py \
    -model_type \
    b \
    -api_token \
    XXX \
    -model_name_or_path \
    gpt-3.5-turbo \
    -device \
    cuda:0 \
    -dataset_name \
    tofu \
    -skip_gen \
    1 \
    -batch_size \
    1 \
    -seed \
    42 \
    -use_small \
    0 \
    -use_rouge \
    1 \
    -use_bart \
    0 \
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
    black_four_full_rhum3_def35both
