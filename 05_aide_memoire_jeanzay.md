# 05 — Aide-mémoire Jean Zay : lancer une config DocWorkflow

Partie A : la marche générale, valable pour toute config (train ou predict).
Partie B : le run LOSTMA (`configs/lostma/jeanzay_lostma_4B.yml`, inférence).
Partie C : le multi-ligne MEDUSA 0.2 k=4 (train puis predict).

## A. Marche générale

### A.0 Repères

| Quoi | Valeur |
|---|---|
| Connexion | `ssh ugg16ol@jean-zay.idris.fr` |
| Compte / GPU | `uoe@h100`, une H100 80 Go par job |
| Code (worktree) | `$WORK/DocWorkflow_ml` (branche `ml-theo`, qui suit `origin/theo-workspace`) |
| Environnement Python | `$WORK/envs/ml-train`, à activer **uniquement** via `source $WORK/DocWorkflow_ml/scripts/jz_env.sh` |
| Données | `$SCRATCH/data/...` (archives zip dans `$STORE`) |
| Modèles HF | `$SCRATCH/models/Medusa0.2Line-{4B,9B}` = `/lustre/fsn1/projects/rech/uoe/ugg16ol/models/...` |
| Dossier de lancement | `$SCRATCH/runs` : logs slurm, `results/`, `outputs_*/` |
| Script générique | `scripts/jz_run.slurm CONFIG <commande docworkflow>` |

Les espaces disque :

- `$WORK` (`/lustre/fswork/...`) : pas de purge. On y met le code et l'environnement.
- `$SCRATCH` (`/lustre/fsn1/...`) : rapide, mais **chaque fichier non lu depuis 30 jours est supprimé**. Tout ce qui s'y trouve doit pouvoir être reconstruit.
- `$STORE` : archive durable, avec peu d'inodes (100k). On y met des **zip ou tar**, jamais des milliers de petits fichiers.

Les QoS H100 :

| QoS | Durée max | Usage |
|---|---|---|
| `qos_gpu_h100-dev` | 2 h | tests |
| `qos_gpu_h100-t3` | 20 h | défaut |
| `qos_gpu_h100-t4` | 100 h | entraînements longs |

Les nœuds de calcul **n'ont pas accès à internet**. Tout ce qui se télécharge (pip, modèles, poids) se fait depuis le nœud de login, avant le job. `jz_env.sh` passe HF et wandb en mode offline.

Une config Jean Zay respecte deux règles de chemins :

- **Données** : chemin relatif `../data/...`. Comme on lance depuis `$SCRATCH/runs`, `../data` désigne `$SCRATCH/data`.
- **Modèles et poids** : chemin **absolu** (`/lustre/...`). Un chemin relatif comme `src/tasks/...` serait cherché dans `$SCRATCH/runs` et ne serait pas trouvé. Les variables `$WORK` et `$SCRATCH` ne sont pas développées dans le YAML : écrire le chemin complet.

### A.1 En local : envoyer le code et les fichiers

```bash
# Code et configs : par git
git add <fichiers> && git commit -m "..." && git push origin theo-workspace

# Ce que git ne transporte pas (gitignoré ou trop gros) : par scp/rsync
rsync -av --progress <fichier_ou_dossier> ugg16ol@jean-zay.idris.fr:<chemin_absolu_sur_JZ>/
```

### A.2 Sur Jean Zay (login) : mettre le code à jour

```bash
cd $WORK/DocWorkflow_ml
git fetch origin && git merge --ff-only origin/theo-workspace
git log --oneline -1          # vérifier qu'on est bien sur le dernier commit
```

### A.3 Données

Pour des données venant du local, envoyer le zip (un seul fichier, plus rapide que des milliers d'images). Ensuite :

```bash
cp <x>.zip $STORE/            # copie durable
mkdir -p $SCRATCH/data/<DATASET> && cd $SCRATCH/data/<DATASET>
unzip -l <x>.zip | head       # regarder le dossier racine du zip pour éviter data/X/X/
unzip -q <x>.zip
```

Pour des données déjà dans `$STORE`, il suffit de les dézipper au même endroit. Si `$SCRATCH` a été purgé, on refait seulement cette étape.

### A.4 Vérifier l'environnement (login, 10 s)

```bash
source $WORK/DocWorkflow_ml/scripts/jz_env.sh
python -c "import torch, transformers, src; print(torch.__version__, transformers.__version__, list(src.__path__))"
```

Attendu : `2.10.0`, `5.2.0`, et un chemin vers `$WORK/DocWorkflow_ml/src`. Sur le login, il n'y a pas de GPU : `unsloth` ne s'y importe pas, c'est normal.

### A.5 (Optionnel mais conseillé) Test interactif en qos dev

À faire avant un job long, ou quand on ne connaît pas la durée.

```bash
srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:1 --hint=nomultithread \
     --constraint=h100 --account=uoe@h100 --time=00:45:00 --qos=qos_gpu_h100-dev bash
source $WORK/DocWorkflow_ml/scripts/jz_env.sh
cd $SCRATCH/runs
python -u -m src -c $WORK/DocWorkflow_ml/<config>.yml <predict|train> -t <tâche>
# Observer la vitesse (barre tqdm), puis Ctrl-C ; quitter avec exit
```

### A.6 Soumettre le job

```bash
mkdir -p $SCRATCH/runs && cd $SCRATCH/runs     # toujours soumettre d'ici
sbatch --job-name=<nom> --time=HH:MM:SS --qos=qos_gpu_h100-t3 \
       $WORK/DocWorkflow_ml/scripts/jz_run.slurm <config relative au worktree> <commande>
```

Les commandes docworkflow possibles :

| But | Commande |
|---|---|
| Pipeline complet (layout, puis lignes, puis HTR) | `predict -t all` |
| Une seule tâche | `predict -t htr` (ou `layout`, `line`) |
| Prédire sur `valid` au lieu de `test` | ajouter `-d valid` |
| Entraîner | `train -t htr` (ou `layout`, `line`) |

Pour enchaîner deux jobs (le second ne démarre que si le premier réussit) :

```bash
J=$(sbatch --parsable <options> $WORK/DocWorkflow_ml/scripts/jz_run.slurm <cfg1> train -t htr)
sbatch --dependency=afterok:$J <options> $WORK/DocWorkflow_ml/scripts/jz_run.slurm <cfg2> predict -t htr
```

Pour choisir `--time` : mesurer en A.5 et ajouter environ 30 % de marge. Un temps trop court tue le job. Un temps trop long allonge l'attente en file.

### A.7 Suivre le job

```bash
squeue --me                    # état : PD = en attente, R = en cours
squeue --me --start            # heure de démarrage estimée
tail -f $SCRATCH/runs/<nom>_<jobid>.out
ssh <nœud> nvidia-smi          # usage GPU (nœud indiqué par squeue)
scancel <jobid>                # annuler un job ; scancel --me pour tous les annuler
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode   # bilan d'un job fini
```

Un job suspendu avec la raison `DependencyNeverSatisfied` attend un job parent qui a échoué. Il faut l'annuler avec `scancel`.

### A.8 Récupérer et archiver les résultats

Les sorties sont dans `$SCRATCH/runs/results/<run_name>/<tâche>/`, et les modèles entraînés dans `$SCRATCH/runs/<model_dir>/`.

```bash
# En local : rapatrier
rsync -av ugg16ol@jean-zay.idris.fr:/lustre/fsn1/projects/rech/uoe/ugg16ol/runs/results/<run_name> ./jz_results/

# Sur Jean Zay : archiver ce qu'on veut garder (un tar, à cause du quota d'inodes de $STORE)
cd $SCRATCH/runs && tar -cf $STORE/<nom>.tar <dossier>
```

### A.9 Pièges connus

- **`ModuleNotFoundError`** : `jz_env.sh` n'a pas été sourcé, ou le paquet manque dans le venv. Pour l'installer : depuis le login, `source jz_env.sh` puis `pip install --no-cache-dir -c $WORK/envs/ml-train.pins.txt <paquet>` (voir B.1 pour le fichier de pins).
- **Le job charge un vieux code** : le worktree n'est pas à jour (A.2). Vérifier `list(src.__path__)` dans le log.
- **`FileNotFoundError` sur des données** : le job n'a pas été soumis depuis `$SCRATCH/runs`, ou les données ont été purgées.
- **Le job reste bloqué ou plante sur un appel réseau** : un chemin de modèle ou de poids n'existe pas, et la bibliothèque tente de le télécharger. Vérifier les chemins absolus.
- **Purge** : un dossier de `$SCRATCH` intact mais vide, c'est la purge. On restaure depuis `$STORE`.
- **Fichiers gitignorés** (poids YOLO `*.pt`, données) : ils ne voyagent pas par git, il faut les copier avec rsync (A.1).

## B. LOSTMA : pipeline complet en inférence

Config : `configs/lostma/jeanzay_lostma_4B.yml`. Elle enchaîne layout YOLO, lignes YOLO et HTR `Medusa0.2Line-4B`, sur `../data/LOSTMA/occ` (environ 900 images, 1,8 Go).

### B.1 Une seule fois : ajouter les dépendances YOLO au venv

Le venv `ml-train` a été créé pour l'entraînement VLM. Il ne contient pas `ultralytics`, `mean-average-precision` ni `scikit-learn`, dont la partie layout/lignes a besoin. Depuis le login :

```bash
source $WORK/DocWorkflow_ml/scripts/jz_env.sh
pip freeze | grep -E '^(torch|torchvision|triton|transformers)==' > $WORK/envs/ml-train.pins.txt
pip install --no-cache-dir -c $WORK/envs/ml-train.pins.txt "ultralytics>=8.4.8" mean-average-precision scikit-learn
python -c "import ultralytics, cv2, mean_average_precision, sklearn; print(ultralytics.__version__)"
pip freeze > $WORK/envs/ml-train.freeze.txt
```

Le fichier de pins empêche pip de remplacer le torch 2.10. Si `import cv2` échoue avec `libGL.so.1` : `pip uninstall -y opencv-python && pip install --no-cache-dir opencv-python-headless`.

### B.2 Envoyer code, poids et données (en local)

```bash
# Code : les configs lostma et les scripts jz_* doivent être commités
git add configs/lostma scripts/jz_env.sh scripts/jz_run.slurm && git commit -m "lostma jeanzay" && git push origin theo-workspace

# Poids YOLO (gitignorés), copiés au même endroit que dans le dépôt
rsync -av src/tasks/layout/models/yolo26s_layout_640px_16bs_50e.pt \
  ugg16ol@jean-zay.idris.fr:/lustre/fswork/projects/rech/uoe/ugg16ol/DocWorkflow_ml/src/tasks/layout/models/
rsync -av src/tasks/line/models/line_det_yolo26s_1280px_8bs_100e_fl0.5_mo0.2.pt \
  ugg16ol@jean-zay.idris.fr:/lustre/fswork/projects/rech/uoe/ugg16ol/DocWorkflow_ml/src/tasks/line/models/

# Données
rsync -av --progress ../data/LOSTMA/occ.zip ugg16ol@jean-zay.idris.fr:/lustre/fsn1/projects/rech/uoe/ugg16ol/data/LOSTMA/
```

Si le dossier `src/tasks/line/models/` (ou `layout`) n'existe pas sur Jean Zay, le créer d'abord avec `mkdir -p`.

### B.3 Sur Jean Zay

```bash
cd $WORK/DocWorkflow_ml && git fetch origin && git merge --ff-only origin/theo-workspace
ls -la src/tasks/layout/models/yolo26s_layout_640px_16bs_50e.pt src/tasks/line/models/line_det_yolo26s_*.pt
ls -laL $SCRATCH/models/Medusa0.2Line-4B/         # config.json + *.safetensors non vides

cd $SCRATCH/data/LOSTMA && cp occ.zip $STORE/ && unzip -q occ.zip    # le zip contient déjà occ/
ls $SCRATCH/data/LOSTMA/occ | head                # 63751 63752 ... (un dossier par volume)

cd $SCRATCH/runs
sbatch --job-name=lostma --time=05:00:00 --qos=qos_gpu_h100-t3 \
       $WORK/DocWorkflow_ml/scripts/jz_run.slurm configs/lostma/jeanzay_lostma_4B.yml predict -t all
```

Les 5 h sont une estimation large, faute de mesure. Pour la mesurer, lancer A.5 avec la même commande et regarder la vitesse de l'étape HTR.

Résultats : `$SCRATCH/runs/results/lostma_occ_jz/{layout,line,htr}/`. L'ALTO final est dans `htr/`. Les rapatrier comme en A.8.

## C. Multi-ligne MEDUSA 0.2, k=4

Configs : `configs/cmmhwr/jeanzay/jeanzay_ML-Medusa0.2-{4B,9B}_k4.yml` (train) et `..._k4_pred.yml` (prédiction sur `valid`). Contexte : `04_setup_jeanzay_multiligne.md`.

### C.1 Vérifications préalables

```bash
ls $SCRATCH/data/ICDAR_Competition
# attendu : gold_nonromance gold_romance platinum_nonromance platinum_romance valid
```

`valid` est indispensable. Le code d'entraînement le cherche à côté des jeux d'entraînement, et la config `_pred` prédit dessus. S'il manque, le restaurer depuis son zip dans `$STORE` (A.3).

```bash
ls -laL $SCRATCH/models/Medusa0.2Line-9B/ $SCRATCH/models/Medusa0.2Line-4B/
du -shL $SCRATCH/models/*        # environ 8 Go (4B) et 18 Go (9B) ; un fichier de 0 octet signifie une purge
```

### C.2 Test dev du 9B (45 min)

Il sert à mesurer la vitesse et la mémoire, et à vérifier l'évaluation, la sauvegarde et la reprise.

```bash
srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:1 --hint=nomultithread \
     --constraint=h100 --account=uoe@h100 --time=00:45:00 --qos=qos_gpu_h100-dev bash
source $WORK/DocWorkflow_ml/scripts/jz_env.sh
cd $SCRATCH/runs
python -c "from unsloth import FastVisionModel; import torch; print(torch.cuda.get_device_name())"

sed -e 's/^run_name: .*/run_name: "devtest-9B"/' \
    -e "s#model_dir: .*#model_dir: 'devtest_9B/'#" \
    -e 's/save_steps: .*/save_steps: 30/' -e 's/eval_steps: .*/eval_steps: 30/' \
    -e 's/logging_steps: .*/logging_steps: 5/' \
    $WORK/DocWorkflow_ml/configs/cmmhwr/jeanzay/jeanzay_ML-Medusa0.2-9B_k4.yml > devtest_9B.yml
python -u -m src -c devtest_9B.yml train -t htr 2>&1 | tee devtest_9B.log
```

Noter quatre valeurs :

- le nombre total de pas (barre `x/NNNN`, environ 4 700 attendus) ;
- les **secondes par pas** (`s/it`, à lire après une vingtaine de pas) ;
- la **durée de l'évaluation** au pas 30 ;
- la mémoire GPU (`ssh <nœud> nvidia-smi` depuis un autre terminal).

Après la création de `devtest_9B/checkpoint-30` : Ctrl-C, relancer la même commande, et vérifier que le run reprend au pas 30. Ensuite, `rm -r devtest_9B devtest_9B.yml`.

En cas d'OOM : `train_batch_size: 8` et `gradient_accumulation_steps: 4` dans la vraie config. Le batch effectif reste à 32. Commiter, pousser, puis refaire A.2.

Le 4B est plus rapide : refaire le même test avec `4B` si on veut un `--time` juste pour lui.

### C.3 Calcul de `--time`

```
--time = (nb_pas × s/pas + 5 × durée_éval + 15 min de chargement) × 1,3
```

Au-delà de 20 h : `qos_gpu_h100-t4`. Autre option : rester en t3 et resoumettre la même commande. L'entraînement reprend seul au dernier `checkpoint-*` de `outputs_ML-Medusa0.2-<taille>_k4/`.

Pour la prédiction, aucune mesure n'existe : 3 h en t3 servent de point de départ.

### C.4 Soumission : train puis predict enchaînés

```bash
cd $SCRATCH/runs
S=$WORK/DocWorkflow_ml/scripts/jz_run.slurm
C=configs/cmmhwr/jeanzay

J9=$(sbatch --parsable --job-name=ML-9B-train --time=<T9> --qos=<qos> $S $C/jeanzay_ML-Medusa0.2-9B_k4.yml train -t htr)
sbatch --dependency=afterok:$J9 --job-name=ML-9B-pred --time=03:00:00 --qos=qos_gpu_h100-t3 \
       $S $C/jeanzay_ML-Medusa0.2-9B_k4_pred.yml predict -t htr

J4=$(sbatch --parsable --job-name=ML-4B-train --time=<T4> --qos=<qos> $S $C/jeanzay_ML-Medusa0.2-4B_k4.yml train -t htr)
sbatch --dependency=afterok:$J4 --job-name=ML-4B-pred --time=03:00:00 --qos=qos_gpu_h100-t3 \
       $S $C/jeanzay_ML-Medusa0.2-4B_k4_pred.yml predict -t htr
```

Le 4B et le 9B tournent en parallèle, chacun sur son GPU.

Si un entraînement dépasse son temps, le job pred dépendant reste en `DependencyNeverSatisfied`. Dans ce cas : `scancel` du job pred, puis resoumettre les deux commandes du modèle concerné. L'entraînement reprend au dernier checkpoint.

### C.5 Après les runs

- Modèles entraînés : `$SCRATCH/runs/outputs_ML-Medusa0.2-{4B,9B}_k4/Medusa0.2Line-{4B,9B}-multiline-k4`. Les archiver : `cd $SCRATCH/runs && tar -cf $STORE/ML-Medusa0.2-9B_k4.tar outputs_ML-Medusa0.2-9B_k4/Medusa0.2Line-9B-multiline-k4`.
- Prédictions sur `valid` : `$SCRATCH/runs/results/ML-Medusa0.2-{4B,9B}_k4_pred/htr/`. Les rapatrier comme en A.8.
- Une fois le résultat jugé bon, supprimer les `checkpoint-*` intermédiaires (plusieurs Go chacun).
