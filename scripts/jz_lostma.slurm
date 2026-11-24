#!/bin/bash
# Pipeline complet LOSTMA (layout YOLO -> lignes YOLO -> HTR Medusa0.2Line-4B) sur 1 H100.
# À soumettre depuis $SCRATCH/runs ; les données doivent être dans $SCRATCH/data/LOSTMA/occ.
#   cd $SCRATCH/runs
#   sbatch --job-name=lostma --time=05:00:00 --qos=qos_gpu_h100-t3 $WORK/DocWorkflow_ml/scripts/jz_lostma.slurm
# Reprise : les sorties déjà produites par volume sont dans results/lostma_occ_jz/.
#SBATCH -C h100
#SBATCH --account=uoe@h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

source $WORK/DocWorkflow_ml/scripts/jz_env.sh
set -eo pipefail
set -x
cd $SCRATCH/runs
python -c "import torch, src; print(torch.__version__, torch.cuda.get_device_name(), list(src.__path__))"
python -u -m src -c $WORK/DocWorkflow_ml/configs/lostma/jeanzay_lostma_4B.yml predict -t all
