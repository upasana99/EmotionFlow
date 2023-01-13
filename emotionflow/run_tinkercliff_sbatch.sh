#!/bin/bash
#SBATCH -J Emotionflow
#SBATCH --account=cxr
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 # this requests 1 node, 1 core. 
#SBATCH --time=1-00:00:00 # 10 minutes
#SBATCH --gres=gpu:1

module reset

module load Anaconda3/2020.11

source activate aml
module reset
which python

# python train.py -tr -wp 0 -bsz 5 -acc_step 1 -lr 1e-4 -ptmlr 1e-5 -dpt 0.3 -bert_path roberta-base -epochs 20 -postfix test-speaker-meld-base-20_bsz_5_acc_1
# python train.py -tr -wp 0 -bsz 1 -acc_step 8 -lr 1e-4 -ptmlr 1e-5 -dpt 0.3 -bert_path roberta-base -epochs 20 -postfix emorynlp-base-epochs=20-acc=8 -tsk emorynlp
python train.py -tr -wp 0 -bsz 1 -acc_step 8 -lr 1e-4 -ptmlr 1e-5 -dpt 0.3 -bert_path roberta-large -epochs 20 -postfix emorynlp-large-epochs=20-acc=8_no_speaker -tsk emorynlp
# python train_daily_dialogue.py -tr -wp 0 -bsz 1 -acc_step 8 -lr 1e-4 -ptmlr 1e-5 -dpt 0.3 -bert_path roberta-base -epochs 20 -postfix daily_dialog-base-epochs20-acc=8_batch=1_all_pts_ignoring_0_macro

# python train_iemocap.py -tr -wp 0 -bsz 1 -acc_step 2 -lr 1e-4 -ptmlr 1e-5 -dpt 0.3 -bert_path roberta-large -epochs 20 -postfix iemocap-large-epochs20-acc=2_nospeaker

exit;