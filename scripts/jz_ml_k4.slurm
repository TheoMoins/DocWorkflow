#!/bin/bash
# Entraînement / prédiction multi-ligne (k=4) MEDUSA 0.2 sur Jean Zay.
# Arguments : SIZE (4B | 9B) et MODE (train | pred). --time et --qos sont passés à sbatch.
# À soumettre depuis $SCRATCH/runs (les chemins ../data/... des configs en dépendent) :
#   cd $SCRATCH/runs
#   J=$(sbatch --parsable --job-name=ML-9B-train --time=HH:MM:SS --qos=qos_gpu_h100-t3 \
#         $WORK/DocWorkflow_ml/scripts/jz_ml_k4.slurm 9B train)
#   sbatch --dependency=afterok:$J --job-name=ML-9B-pred --time=HH:MM:SS --qos=qos_gpu_h100-t3 \
#         $WORK/DocWorkflow_ml/scripts/jz_ml_k4.slurm 9B pred
# Un job train arrêté par la limite de temps reprend au dernier checkpoint-* si on le resoumet.
#SBATCH -C h100
#SBATCH --account=uoe@h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24          # 1/4 des CPU d'un nœud 4-GPU H100
#SBATCH --hint=nomultithread
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

SIZE=$1
MODE=$2
CFG=$WORK/DocWorkflow_ml/configs/cmmhwr/jeanzay/jeanzay_ML-Medusa0.2-${SIZE}_k4

source $WORK/DocWorkflow_ml/scripts/jz_env.sh
set -eo pipefail
set -x

cd $SCRATCH/runs
python -c "import torch, src; print(torch.__version__, torch.cuda.get_device_name(), list(src.__path__))"

case $MODE in
    train) python -u -m src -c ${CFG}.yml train -t htr ;;
    pred)  python -u -m src -c ${CFG}_pred.yml predict -t htr ;;
    *)     echo "MODE inconnu : $MODE (train | pred)"; exit 1 ;;
esac
