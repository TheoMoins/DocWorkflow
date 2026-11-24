# 04 — Setup Jean Zay pour l'entraînement multi-ligne (état présent)

Document de passation : il décrit où en est le projet et ce qui reste à faire pour lancer les entraînements multi-ligne sur Jean Zay. Contexte plus large : `01_synthese_segmentation_ligne.md`, `02_analyse_et_plan_action.md`, `03_plan_htr_multiligne.md` (à lire en cas de doute, sans déborder du scope).

Contraintes de travail : pas de GPU local (le code est préparé et validé sur CPU, l'utilisateur pousse et exécute sur Jean Zay). Ne pas éditer le plan `03` ni committer sans demande explicite.

## 1. Objectif

Fine-tuner les modèles publiés `ENC-PSL/Medusa0.2Line-4B` et `ENC-PSL/Medusa0.2Line-9B` (modèles complets mergés, donc sans `base_model`) sur des données multi-ligne (k=4 lignes par crop), 1 epoch chacun, puis prédire sur le dev set `valid` de la compétition ICDAR.

Raison : le MEDUSA 0.2 zero-shot, entraîné sur lignes isolées, ne renvoie que la première ligne d'un crop de k lignes (100 % de blocs en `[mismatch]`, aucun changement de prompt n'a aidé).
Hors scope pour l'instant : récupérer les sorties T3-1 / T3-2 (purgées), gérer les pages de moins de k lignes (elles sont ignorées à l'entraînement).

## 2. Code (côté local, branche `theo-workspace`)

Le code est prêt et testé (suite complète : 140 passed, 2 skipped). Les changements suivants ne sont **pas committés** et doivent être commit/push avant de les récupérer sur Jean Zay :

- `src/tasks/htr/vlm_multiline_htr.py` : `VLMMultiLineHTR` avec `train()`, `_prepare_training_data_lines` (blocs de k lignes sans franchir un TextBlock, pages < k lignes ignorées), `_format_conversation`, `_process_batch` (écriture ALTO par id de ligne), `save_steps` / `eval_steps` / `logging_steps` lus depuis la config.
- `src/tasks/htr/base_vlm_htr.py` et `src/tasks/htr/vlm_line_htr.py` : `_prepare_samples_with_conventions` déplacée dans la base.
- `src/alto/alto_lines.py` : `read_lines_geometry` expose `block_id`.
- `scripts/merge_lora_medusa.py` : accepte `BASE LORA OUTPUT` en arguments.
- `tests/test_tasks/test_vlm_multiline_htr.py`.
- Configs : `configs/cmmhwr/jeanzay/` (suppression de `jeanzay_ML-T3-2_k4(.pred).yml` et `jeanzay_ML-T3-3_k4(.pred).yml`, ajout des quatre fichiers ci-dessous).

Configs Jean Zay (dans `configs/cmmhwr/jeanzay/`) :

| Fichier | Rôle |
|---|---|
| `jeanzay_ML-Medusa0.2-4B_k4.yml` | entraînement 4B |
| `jeanzay_ML-Medusa0.2-4B_k4_pred.yml` | prédiction 4B sur `valid` |
| `jeanzay_ML-Medusa0.2-9B_k4.yml` | entraînement 9B |
| `jeanzay_ML-Medusa0.2-9B_k4_pred.yml` | prédiction 9B sur `valid` |

Paramètres clés : type `VLMMultiLineHTR`, `lines_per_crop: 4`, `max_new_tokens: 512`, `line_batch_size: 4`, `max_pixels: 401408`, `lora_r: 64`, `max_seq_length: 1024`, `train_batch_size: 16`, `gradient_accumulation_steps: 2`, `epochs: 1`, `save_steps` / `eval_steps: 1000`, `logging_steps: 50`, lr 5e-5, `model_dir: outputs_ML-Medusa0.2-{4B,9B}_k4/`.
Données d'entraînement : `platinum_romance`, `platinum_nonromance`, `gold_romance`, `gold_nonromance` ; validation : `valid` ; chemins relatifs `../data/ICDAR_Competition/...`.
Chemins de modèles : `model_name` (train) et `base_model` (pred) = `/lustre/fsn1/projects/rech/uoe/ugg16ol/models/Medusa0.2Line-{4B,9B}` (chemins à créer, voir §4). Le modèle entraîné est sauvé dans `outputs_ML-Medusa0.2-{size}_k4/Medusa0.2Line-{size}-multiline-k4`, et la config `_pred` pointe dessus.
Une étape de ~4 700 pas par modèle est attendue ; `--time` du job slurm à fixer d'après les secondes/pas mesurées (× 1,3 + temps de prédiction).

## 3. Jean Zay : faits

- Compte `uoe@h100` ; GPU H100 ; modules `arch/h100` et `pytorch-gpu/py3/2.8.0` (Python 3.12) ; QoS : dev ≤ 2 h, t3 ≤ 20 h, t4 ≤ 100 h.
- Les nœuds de calcul n'ont pas d'accès internet ; l'accès réseau depuis les nœuds de login (pip, HF) est à vérifier.
- `$SCRATCH` = `/lustre/fsn1/projects/rech/uoe/ugg16ol` : **purgé après 30 jours sans accès**, fichier par fichier (des dossiers peuvent survivre vidés).
- `$WORK` = `/lustre/fswork/projects/rech/uoe/ugg16ol` ; `$STORE` : archivage durable.
- Le dépôt `DocWorkflow` sur Jean Zay est sur la branche `pagelevel`, qui a divergé de `theo-workspace` (`train()` et `_prepare_samples_with_conventions` y sont dans `BaseVLMHTR`, `_prepare_training_data_lines` y est renommée `_prepare_training_data`). Fusionner est un vrai refactor : ne pas le faire ; utiliser un worktree séparé sur `origin/theo-workspace` (par exemple `git worktree add -b ml-theo <chemin> origin/theo-workspace`).
- Le paquet `src` est un paquet sans `__init__.py` (namespace) : `src.__file__` vaut `None`, vérifier avec `list(src.__path__)`.
- Une installation éditable historique pointe sur `$SCRATCH/DocWorkflow` : il faut forcer `PYTHONPATH` sur le worktree utilisé.

## 4. État constaté sur Jean Zay

**Modèles** : le cache HF global a été copié dans `$STORE` (copie durable) et `$SCRATCH/cache/huggingface/hub/` (contient `models--ENC-PSL--Medusa0.2Line-4B` et `-9B`). À faire : créer les liens

```bash
mkdir -p $SCRATCH/models
ln -s $SCRATCH/cache/huggingface/hub/models--ENC-PSL--Medusa0.2Line-4B/snapshots/<hash4B> $SCRATCH/models/Medusa0.2Line-4B
ln -s $SCRATCH/cache/huggingface/hub/models--ENC-PSL--Medusa0.2Line-9B/snapshots/<hash9B> $SCRATCH/models/Medusa0.2Line-9B
```
après avoir vérifié (`ls -laL`, `du -shL`) que les fichiers (`config.json`, `*.safetensors`) sont présents et non vides (non vérifié à ce jour). Pas de copie dans `$WORK`. Ajouter `export HF_HUB_OFFLINE=1` dans les scripts slurm.

**Données** : `$SCRATCH/data` est vide. Les zips des données sont dans `$STORE` (non listés ici). À faire : `unzip -l` pour voir la structure, puis restaurer dans `$SCRATCH/ml/data/ICDAR_Competition` en évitant un double dossier. Les chemins relatifs `../data/ICDAR_Competition/...` des configs supposent que le répertoire de travail est un sous-dossier frère de `data`.

**Worktree** : un worktree existe dans `/lustre/fswork/projects/rech/uoe/ugg16ol/DocWorkflow_ml` (créé par erreur dans `$WORK`) ; `python -c "import src; print(list(src.__path__))"` renvoie ce chemin. `$SCRATCH/ml/DocWorkflow_ml` n'a pas été vérifié (existence inconnue). Choisir un seul emplacement : le code pèse peu et `$WORK` n'est pas purgé, donc garder le worktree dans `$WORK` est acceptable, avec données et sorties dans `$SCRATCH`.

**Environnement Python : cassé (point bloquant).** `$SCRATCH/DocWorkflow/envs/vlm-training` (ancien venv) a été vidé par la purge :
- `bin/` ne contient que des liens `python`, `python3`, `python3.12` vers le Python du module (pas d'`activate`, pas de `pip`, pas de `pyvenv.cfg`).
- `lib/python3.12/site-packages` : 26 Mo, 306 entrées, mais **aucun fichier `.py`** (ni `__init__.py`) : seuls des dossiers vides et les `*.dist-info` restent. Résultat : Python ne voit plus ces dossiers comme des paquets, utilise ceux du module, et `import transformers` échoue (`torch.__version__` vaut `None`, mélange de métadonnées torch 2.10.0 de l'ancien env et torch 2.8.0 du module).
- Versions lues dans les `dist-info` : `torch 2.10.0`, `transformers 5.2.0`, `trl 0.24.0`, `peft 0.18.1`, `unsloth 2026.3.4`, `unsloth_zoo 2026.3.2`, `bitsandbytes 0.49.2`, `datasets 4.3.0`, `accelerate 1.13.0`.
- On ne sait pas si l'ancien env embarquait vraiment son propre torch 2.10.0 (et les libs CUDA) par-dessus le module, ni comment il était activé dans les scripts slurm.
- Le premier exemple de slurm que j'avais donné utilisait `source envs/vlm-training/bin/activate` : c'est une hypothèse tirée d'un exemple, jamais validée.

Une question a été envoyée à la collègue qui avait mis en place l'env initial (comment il a été créé, torch 2.10 propre ou du module, commande d'activation, parade à la purge, copies des données / sorties T3). Consulter sa réponse d'abord si elle est disponible.

## 5. Étapes restantes, dans l'ordre

1. Commit / push des changements locaux (§2).
2. Remettre en état l'environnement Python (voir §4).
3. Créer les liens de modèles et vérifier leur contenu (§4).
4. Restaurer les données depuis `$STORE` (§4).
5. Mettre le code à jour dans le worktree (`git fetch` puis fast-forward / nouveau worktree sur `origin/theo-workspace`) et vérifier l'import avec `list(src.__path__)`.
6. Test interactif court en qos dev (45 min, 9B d'abord) pour mesurer les secondes/pas et vérifier que le chargement du modèle et le chargement des données fonctionnent.
7. Scripts slurm pour le 4B et le 9B : entraînement puis prédiction avec la config `_pred`, `--time` d'après la mesure du pas 6.
8. Plus tard (non demandé) : scorer le CER à k=4 contre MEDUSA ligne par ligne et contre le zero-shot de la phase 1 (jamais mesuré proprement), phases 3 à 5 du plan, réconciliation des refactors `pagelevel` / `theo-workspace`.

## 6. Documentation Jean Zay à fournir à la prochaine conversation

L'utilisateur doit donner le lien de la documentation Jean Zay (environnements Python / modules, pip, slurm, stockage) pour choisir la bonne méthode de création de l'env (venv sur module ou environnement dédié) et la bonne gestion de la purge.
