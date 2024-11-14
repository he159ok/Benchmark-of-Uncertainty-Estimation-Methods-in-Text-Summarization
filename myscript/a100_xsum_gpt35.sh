#!/bin/bash

#SBATCH --account=stgnn
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --time=6-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -J f_xsum_gpt
#SBATCH -o ./log/f_xsum_gpt.output.txt
#SBATCH -e ./log/f_xsum_gpt.error.txt


#module spider anaconda
#module load Anaconda3/2020.11
#source activate ~/env/unc_kg

# assist 789
# samle 789
#

dname=$(dirname "$PWD")
cd $dname
echo $dname

CUDA_VISIBLE_DEVICES=0,

python main_ats_revise.py \
    -model_type \
    b \
    -api_token \
    XXX \
    -model_name_or_path \
    gpt-3.5-turbo \
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
    -use_claude \
    0 \
    -use_unieval_overall \
    1 \
    -ue_cal_name \
    black_four_full
