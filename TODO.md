# TODO — Résolution des conflits d'environnements

## Corrections déjà appliquées

- [x] `src/tasks/htr/base_vlm_htr.py` — suppression du guard unsloth dans `load()` et code mort (`elif False:`)
- [x] `src/alto/alto_lines.py` — `XMLPage` (kraken) remplacé par un parser `lxml` natif ; kraken retiré des imports ALTO
- [x] `scripts/setup_envs.sh` — `kraken`, `rich`, `iso639` supprimés de la section vlm-training

---

## Étape 1 — Vérifier que les corrections ne cassent rien (30 min)

Avant de toucher aux envs, valider dans les envs actuels :

```bash
# Dans vlm-training
source envs/vlm-training/bin/activate
python -c "from src.alto.alto_lines import read_lines_geometry; print('OK')"
pytest tests/test_alto/ -v

# Dans main
source envs/main/bin/activate
pytest tests/ -v
```

---

## Étape 2 — Corriger `pyproject.toml` (1h)

- [x] Ajouter `peft>=0.10.0` aux dépendances principales (nécessaire pour l'inférence VLM avec LoRA)
- [x] Supprimer ou élargir les upper bounds incorrects :
  ```toml
  # Avant → Après
  "torch>=2.1.0,<2.5.0"      → retiré (géré par feature pixi)
  "torchvision>=0.19.0,<0.24.0" → retiré (géré par feature pixi)
  "pandas>=2.0.0,<2.3.0"     → "pandas>=2.0.0"
  "matplotlib>=3.7.0,<3.9.0" → "matplotlib>=3.7.0"
  "kraken>=5.3.0,<6.0.0"     → "kraken>=5.3.0"
  ```

---

## Étape 3 — Créer `pixi.toml` à la racine (2-3h)

Le template de base est dans `KNOWN_ISSUES.md`. Points d'attention :

- [x] **torch/torchvision** dans `feature.main` : pytorch=2.4.*, torchvision=0.19.* via channel pytorch → couplage résolu automatiquement (torch 2.4.1 + torchvision 0.19.1). Dans `feature.train`, pytorch>=2.5.0.
- [x] **yaltai** : pixi 0.56 ne supporte pas `no-deps` PyPI → sorti de pypi-dependencies, installé via `pixi run install-yaltai` (pip --no-deps yaltai==2.0.5 fast-deskew==1.0)
- [x] **peft** : dans les deux features
- [x] **kraken** : dans `feature.main` uniquement
- [x] **scikit-learn** ajouté aux deps partagées (utilisé par src/utils/utils.py)
- [x] **fsspec** contraint à `<2026.3` dans feature.train (2026.3+ a supprimé l'extra `http` requis par datasets)

Générer le lock et le committer :

```bash
pixi install -e main
pixi install -e train
git add pixi.lock
git commit -m "add pixi.lock"
```

Le lock committé est la garantie de reproductibilité.

---

## Étape 4 — Valider les envs pixi (1h) ✅

```bash
# Imports
pixi run -e main python -c "import torch, torchvision, ultralytics, kraken, yaltai, peft; print('main OK')"
pixi run -e train python -c "from unsloth import FastVisionModel; import trl, peft; print('train OK')"

# Tests
pixi run -e main pytest tests/ -v
pixi run -e train python -c "from src.alto.alto_lines import read_lines_geometry; print('OK')"

# CLI
pixi run -e main docworkflow --help

# Couplage torch/torchvision (créer ce script)
pixi run -e main python scripts/check_versions.py
pixi run -e train python scripts/check_versions.py
```

`scripts/check_versions.py` à créer — vérifie que les versions torch et torchvision sont bien couplées :

```python
import torch, torchvision
tv = torchvision.__version__
t = torch.__version__
tv_minor = int(tv.split(".")[1])
t_minor = int(t.split(".")[1])
assert t_minor == tv_minor - 15, f"torchvision {tv} incompatible avec torch {t}"
print(f"torch {t} + torchvision {tv} : OK")
```

---

## Étape 5 — Ajouter les marqueurs de test (30 min) ✅

- [x] Dans `pytest.ini`, ajouter `requires_main` et `requires_training`
- [x] Taguer `test_yaltai_imports` et `test_kraken_imports` avec `@pytest.mark.requires_main`
- [x] Créer `tests/test_imports_train.py` (avec skip automatique si pas de GPU) :
  - `test_unsloth_import` — skippé si pas de GPU/unsloth
  - `test_training_stack` — skippé si pas de GPU/unsloth
  - `test_alto_lines_in_train_env` — passe dans les deux envs (lxml only)

Résultat final pixi run -e main pytest : **16 failed / 57 passed / 2 skipped**
(les 16 échecs sont pré-existants, non liés à la migration pixi — voir ci-dessous)

### Tests pré-existants à corriger séparément

| Fichier | Raison |
|---|---|
| `test_alto_lines.py::test_extract_lines_from_alto` | parser lxml ne retourne pas de lignes (bug Étape 1) |
| `test_alto_lines.py::test_convert_lines_to_boxes` | idem |
| `test_base_tasks.py` (3 tests) | `DummyTask` ne satisfait plus l'interface abstraite (`_process_batch`, `_score_batch`) |
| `test_kraken_htr.py` (5 tests) | API changée : nom de tâche, méthode supprimée, structure dict retournée, messages d'erreur |
| `test_kraken_line.py` (3 tests) | `ModuleNotFoundError` sur le mock `kraken_line.glob` (conflit de nommage) |
| `test_yolo_layout.py` (3 tests) | Nom de tâche changé, structure gt dir, message d'erreur |

- [ ] Créer `tests/test_imports_train.py` :
  ```python
  import pytest

  @pytest.mark.requires_training
  def test_unsloth_import():
      from unsloth import FastVisionModel
      assert FastVisionModel is not None

  @pytest.mark.requires_training
  def test_training_stack():
      import trl, peft, accelerate, bitsandbytes, datasets
  ```

---

## Ordre de priorité

1. **Étape 1** — valider les corrections déjà faites (tests ALTO + tests main)
2. **Étape 2** — corrections rapides sur `pyproject.toml`, débloque peft/VLM
3. **Étape 3** — migration pixi, le plus long mais le fix structurel
4. **Étape 4** — validation complète des deux envs pixi
5. **Étape 5** — outillage test pour maintenir ça dans le temps
