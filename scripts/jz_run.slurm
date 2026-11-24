#!/bin/bash
# Lance n'importe quelle commande docworkflow sur 1 H100 de Jean Zay.
# Arguments : CONFIG (chemin relatif au worktree, ou absolu) puis la commande docworkflow.
# --job-name, --time et --qos sont passés à sbatch. À soumettre depuis $SCRATCH/runs :
#   cd $SCRATCH/runs
#   sbatch --job-name=lostma --time=05:00:00 --qos=qos_gpu_h100-t3 \
#       $WORK/DocWorkflow_ml/scripts/jz_run.slurm configs/lostma/jeanzay_lostma_4B.yml predict -t all
#SBATCH -C h100
#SBATCH --account=uoe@h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24          # 1/4 des CPU d'un nœud 4-GPU H100
#SBATCH --hint=nomultithread
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

CFG=$1
shift
[[ $CFG = /* ]] || CFG=$WORK/DocWorkflow_ml/$CFG

source $WORK/DocWorkflow_ml/scripts/jz_env.sh
set -eo pipefail
set -x

mkdir -p $SCRATCH/runs
cd $SCRATCH/runs
python -c "import torch, src; print(torch.__version__, torch.cuda.get_device_name(), list(src.__path__))"
python -u -m src -c $CFG "$@"
