#!/bin/bash
#SBATCH -p meta_gpu-x # partition (queue)
#SBATCH -t 20-00:00 # time (D-HH:MM)
#SBATCH -c 5 # number of cores
#SBATCH -a 1-13 # array size
#SBATCH --gres=gpu:1
#SBATCH -D /home/ahnj/repo/autodl/AutoDL/ # Change working_dir
#SBATCH -o /home/ahnj/repo/autodl/AutoDL/log/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /home/ahnj/repo/autodl/AutoDL/log/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID

echo "source activate"
source activate ml4aad
echo "run script"
python bohb_epoch.py $SLURM_ARRAY_TASK_ID 13
echo "done"
