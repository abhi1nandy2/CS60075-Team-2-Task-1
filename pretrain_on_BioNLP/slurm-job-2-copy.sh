#!/bin/bash

#SBATCH -J pretrain_task1            #Job name(--job-name)
#SBATCH -o logs/slurm_%j.log          #Name of stdout output file(--output)
#SBATCH -e logs/slurm_%j.log   #Name of stderr error file(--error)
#SBATCH -p gpu              #Queue (--partition) name
#SBATCH -n 1                    #Total Number of mpi tasks (--ntasks .should be 1 for serial)
#SBATCH --gres=gpu:2
#SBATCH -c 8                    #(--cpus-per-task) Number of Threads

#SBATCH --mail-user=raj12345@iitkgp.ac.in        # user's email ID where job status info will be sent
#SBATCH --mail-type=ALL        # Send Mail for all type of event regarding the job
#SBATCH --time 3-0            # 3 days max

module load compiler/intel-mpi/mpi-2019-v5
module load compiler/cudnn/7.6.2
module load compiler/cuda/10.1

#source /home/$USER/.bashrc
# source /home/$USER/.bash_aliases

export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi
python --version
nvcc --version
python pretrain_manuals.py --from_pretrained --per_dev_batch_size 8 --logging_dir runs #--checkpoint_path embert_model_from_pretrained/checkpoint-458000
#python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 pretrain_manuals.py --from_pretrained --layerwise_lr_decay --per_dev_batch_size 8
