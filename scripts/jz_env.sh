module purge
module load arch/h100 python/3.12.<x>      # même module qu'à la création
source $WORK/envs/ml-train/bin/activate
export PYTHONNOUSERSITE=1                   # ignore ~/.local (anciennes installs --user)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline                   # pas d'internet sur les nœuds de calcul
export TRITON_CACHE_DIR=${JOBSCRATCH:-$SCRATCH/cache}/triton
