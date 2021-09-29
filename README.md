# anomaly_transformer

To train, use at minimum 2 GPUs on Compute Canada (uses PyTorch distributed training)
To test, can use single GPU (no use of distributed data during testing)

# example training script:
#!/bin/bash
#SBATCH -A def-yalda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --mail-user=YOUREMAIL@uwo.ca
#SBATCH --mail-type=ALL
module load python/3.7
source ../transformers/env/bin/activate --DIRECTORY OF VENV

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

python3 ddp_mvtec_trans_anom.py -ma $master_addr -g 4 --train_cls "bottle"

deactivate

# Parameters
-ma --master_addr || See example script on how to define master_addr
-n --nodes || number of nodes to use, keep at 1 for now
-g --gpus || number of GPUs to use. Match your Compute Canada Script.
-nr --nr || keep default for now, ranking within single node
--start_epoch || epoch to start training at
--epochs || how many epochs to run
--resume_enc || 
--resume_dec || 
--train || 'yes' if training, 'no' if testing
--checkpoint_directory || directory where checkpoints are located (for testing)
--test_cls || single class to test. For example, 'bottle' on mvtec
--train_cls || list of classes to test. For example, 'bottle' 'capsule' 'toothbrush' on mvtec

# Requirements
Sorry i dont have a requirements file yet. I will list what I can here
PIL, torchvision, torch, matplotlib, numpy, cv2, glob, einops
