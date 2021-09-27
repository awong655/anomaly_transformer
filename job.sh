#!/bin/bash
#SBATCH -A def-yalda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --mail-user=awong655@uwo.ca
#SBATCH --mail-type=ALL
module load python/3.7
source ../transformers/env/bin/activate

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

#python3 analysis.py --checkpoint_directory ./checkpoint --data_directory ./data --save_mask_tensor True --get_last_attn True
#python3 ddp_mvtec_trans_anom.py -ma $master_addr -g 4 --train_cls "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper"
python3 ddp_mvtec_trans_anom.py -ma $master_addr -g 4 --train_cls "bottle"
#python3 ddp_train_mvtec_new.py -j 2 --epochs 100 -b 70 --lr 0.1 --wd 5e-4 --world-size 2

deactivate

