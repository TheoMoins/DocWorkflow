#!/bin/bash
#
# train_line_sweep.sh — balayage d'entraînement YOLO pour la segmentation de
# ligne, puis évaluation de chaque checkpoint sur le corpus de test gelé.
#
# Complète scripts/benchmark_line_models.sh, qui ne fait que predict+score sur
# des modèles déjà entraînés : ici on entraîne, et on évalue dans le même
# mouvement. Le benchmark n'est pas modifié.
#
#   0  référence corrigée      yolo26s, détection, 640 px, 50e, fliplr=0, mosaic=0
#   1  résolution              img_size ∈ {640, 1280, 1536}
#   2  augmentations           fliplr ∈ {0, 0.5} × mosaic ∈ {0.0, 0.2, 1.0}
#   3  taille de modèle        yolo26n vs yolo26s
#   4  époques                 {30, 50, 100}
#   5  formulation             détection vs segmentation, config par ailleurs identique
#
# Les noms de run sont une fonction déterministe des hyperparamètres, et un run
# dont les poids existent déjà n'est pas réentraîné : les recouvrements entre
# étages (640 px apparaît aux étages 0, 1, 2… ; 50 époques aux étages 0 à 3) ne
# coûtent rien. Le balayage complet fait ainsi 12 entraînements, pas 6+3+6+2+3+1.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   ./scripts/train_line_sweep.sh [options]
#
#   --stages "0 1 2"     étages à exécuter (défaut : "0 1 2 3 4 5"). Chaque étage
#                        repart du gagnant du précédent, mesuré sur le corpus gelé.
#   --device DEV          device d'entraînement : cpu, cuda, cuda:0 (défaut : cuda:0).
#   --eval-device DEV     device de predict+score : cpu, cuda, cuda:0 (défaut :
#                        cpu — ce sont les chiffres de la phase A, directement
#                        comparables ; `--eval-device cuda` va beaucoup plus vite).
#                        Un index nu (`0`) est accepté pour les deux options et
#                        réécrit en `cuda:0` — YoloLineTask.load("pretrained")
#                        fait un torch `model.to()` **avant** d'appeler
#                        `model.train()`, et torch.device("0") lève « Invalid
#                        device string » là où ultralytics l'aurait accepté.
#                        Limite connue : le multi-GPU (`0,1`) casse au même
#                        endroit, pour la même raison, et n'est pas réécrit ici —
#                        ce n'est pas ce script mais YoloLineTask.to_device() qui
#                        en aurait besoin.
#   --data PATH          corpus de test gelé (défaut : ../data/ICDAR_CMMHWR_original_data)
#   --dataset-root PATH  dossier contenant les datasets YOLO
#                        (défaut : ../data/data_yolo — à changer sur un serveur
#                        où l'arborescence diffère, p. ex. /data/theo/WPC1/data/data_yolo)
#   --batch-size N       batch d'entraînement (défaut : 16)
#   --batch-map "..."    batch par résolution, p. ex. "640:16 1280:8 1536:4".
#                        À utiliser en cas d'OOM à haute résolution. Le biais
#                        introduit reste limité : ultralytics accumule les
#                        gradients jusqu'à nbs=64, donc le batch *optimiseur*
#                        reste 64 tant que batch ≤ 64. À signaler dans le rapport.
#   --epochs N           époques de référence pour les étages 0-3 (défaut : 50)
#   --seed N             graine (défaut : 42)
#   --select-metric COL  colonne de results.csv qui départage les étages
#                        (défaut : dataset_test/map50-95)
#   --fix-*              force une valeur au lieu de la faire élire :
#                        --fix-img-size, --fix-fliplr, --fix-mosaic,
#                        --fix-model, --fix-epochs. Sert à reprendre un balayage
#                        interrompu sans rejouer les étages précédents.
#   --retrain            réentraîne même si les poids existent déjà
#   --rescore            rejoue predict+score même si results.csv existe
#   --no-wandb           désactive la trace wandb
#   --dry-run            affiche les runs et les configs générées, n'exécute rien
#
set -uo pipefail

# Le récapitulatif est un CSV, et awk formate les nombres selon la locale : sous
# une locale française, `printf "%.2f"` sort « 6,38 ». Cette virgule injectait un
# champ supplémentaire dans summary.csv et décalait d'une colonne toutes les
# métriques qui suivaient — un tableau faux, sans rien d'anormal à l'œil. La
# comparaison numérique des métriques en dépend aussi.
export LC_NUMERIC=C

on_interrupt() {
    echo ""
    echo "Interrompu. Les poids et les scores déjà produits sont conservés :"
    echo "relancer la même commande reprend là où elle s'est arrêtée."
    exit 130
}
trap on_interrupt INT TERM

cd "$(dirname "$0")/.."

MODELS_DIR="src/tasks/line/models"
TEMPLATE_TRAIN="configs/line/yolo_line_train.yml"
TEMPLATE_EVAL="configs/line/yolo_line.yml"
GEN_DIR="configs/line/generated"
TRAIN_PROJECT="LS-training"

DET_DATASET="medieval-segmentation-yolo_lines"
SEG_DATASET="medieval-segmentation-yolo_lines_polygon"

DATA="../data/ICDAR_CMMHWR_original_data"
DATASET_ROOT="../data/data_yolo"
DEVICE="0"
EVAL_DEVICE="cpu"
BATCH_SIZE="16"
BATCH_MAP=""
BASE_EPOCHS="50"
SEED="42"
SELECT_METRIC="dataset_test/map50-95"
STAGES="0 1 2 3 4 5"
USE_WANDB="true"
RETRAIN="false"
RESCORE="false"
DRY_RUN="false"

# Valeurs courantes de l'ablation. Les étages les réécrivent au fur et à mesure ;
# --fix-* les impose et empêche l'élection.
CUR_IMGSZ="640"
CUR_FLIPLR="0.0"
CUR_MOSAIC="0.0"
CUR_MODEL="yolo26s"
CUR_EPOCHS=""
FIXED_IMGSZ="false"; FIXED_FLIPLR="false"; FIXED_MOSAIC="false"
FIXED_MODEL="false"; FIXED_EPOCHS="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages)        STAGES="$2"; shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --eval-device)   EVAL_DEVICE="$2"; shift 2 ;;
        --data)          DATA="$2"; shift 2 ;;
        --dataset-root)  DATASET_ROOT="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        --batch-map)     BATCH_MAP="$2"; shift 2 ;;
        --epochs)        BASE_EPOCHS="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --select-metric) SELECT_METRIC="$2"; shift 2 ;;
        --fix-img-size)  CUR_IMGSZ="$2"; FIXED_IMGSZ="true"; shift 2 ;;
        --fix-fliplr)    CUR_FLIPLR="$2"; FIXED_FLIPLR="true"; shift 2 ;;
        --fix-mosaic)    CUR_MOSAIC="$2"; FIXED_MOSAIC="true"; shift 2 ;;
        --fix-model)     CUR_MODEL="$2"; FIXED_MODEL="true"; shift 2 ;;
        --fix-epochs)    CUR_EPOCHS="$2"; FIXED_EPOCHS="true"; shift 2 ;;
        --retrain)       RETRAIN="true"; shift ;;
        --rescore)       RESCORE="true"; shift ;;
        --no-wandb)      USE_WANDB="false"; shift ;;
        --dry-run)       DRY_RUN="true"; shift ;;
        -h|--help)       sed -n '2,80p' "$0"; exit 0 ;;
        *) echo "Option inconnue : $1" >&2; exit 2 ;;
    esac
done

[[ -n "$CUR_EPOCHS" ]] || CUR_EPOCHS="$BASE_EPOCHS"

# `--device` ET `--eval-device` passent tous les deux, à un moment ou un autre,
# par un torch `model.to()` brut (YoloLineTask.load() le fait dès le chargement
# des poids pré-entraînés, avant même model.train()) — et torch.device("0")
# lève « Invalid device string » là où ultralytics select_device l'accepte. On
# réécrit donc l'index nu en notation torch plutôt que de laisser le balayage
# casser au premier run, après un préflight qui aura semblé tout valider.
normalize_device() {  # normalize_device VALEUR NOM_OPTION
    case "$1" in
        [0-9]|[0-9][0-9])
            echo "  note : --${2} ${1} réécrit en cuda:${1}" >&2
            echo "cuda:${1}"
            ;;
        *) echo "$1" ;;
    esac
}
DEVICE=$(normalize_device "$DEVICE" "device")
EVAL_DEVICE=$(normalize_device "$EVAL_DEVICE" "eval-device")

OUTPUT_DIR=$(grep -m1 '^output_dir:' "$TEMPLATE_EVAL" | sed 's/output_dir: *"\?\([^"]*\)"\?.*/\1/')
OUTPUT_DIR="${OUTPUT_DIR:-results}"
STAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${OUTPUT_DIR}/line_sweep_${STAMP}"
LOG_DIR="${SWEEP_DIR}/logs"
SUMMARY="${SWEEP_DIR}/summary.csv"
# Registre de durées **persistant**, partagé par tous les balayages. Un run
# réutilisé n'est pas réentraîné et ne produirait donc aucune durée : sans ce
# registre, la colonne min/époque du récapitulatif se viderait dès la première
# reprise, et c'est précisément le chiffre qui sert à dimensionner la suite.
TIMINGS="${OUTPUT_DIR}/line_sweep_timings.csv"

mkdir -p "$GEN_DIR" "$LOG_DIR"

# ── Helpers CSV ───────────────────────────────────────────────────────────────
# results.csv est « large » (une ligne d'en-têtes, une ligne de valeurs) et
# results_zonemap.csv est « long » (metric,value,description). D'où deux
# lecteurs. Tous deux cherchent par NOM de colonne : un jeu de métriques qui
# change d'un run à l'autre ne doit pas décaler silencieusement le tableau.
csv_get() {  # csv_get FICHIER COLONNE
    local f="$1" col="$2"
    [[ -f "$f" ]] || return 1
    awk -v want="$col" -F',' '
        NR==1 { for (i=1;i<=NF;i++) if ($i==want) c=i; next }
        NR==2 { if (c) print $c; exit }
    ' "$f"
}

kv_get() {  # kv_get FICHIER CLE  (results_zonemap.csv)
    local f="$1" key="$2"
    [[ -f "$f" ]] || return 1
    awk -v want="$key" -F',' '$1==want { print $2; exit }' "$f"
}

fmt() {  # arrondit pour l'affichage, laisse passer le vide
    local v="${1:-}"
    [[ -n "$v" ]] || { echo "-"; return; }
    awk -v v="$v" 'BEGIN{ if (v+0==v) printf "%.4f", v; else print v }'
}

# ── Dérivations ───────────────────────────────────────────────────────────────
batch_for() {  # batch_for IMGSZ — --batch-map l'emporte sur --batch-size
    local imgsz="$1" entry
    for entry in $BATCH_MAP; do
        [[ "${entry%%:*}" == "$imgsz" ]] && { echo "${entry##*:}"; return; }
    done
    echo "$BATCH_SIZE"
}

# Nom de run = fonction déterministe des hyperparamètres. C'est ce qui rend le
# balayage reprenable et permet aux étages de se recouvrir sans réentraîner.
run_name_for() {  # run_name_for FORM MODEL IMGSZ BS EPOCHS FLIPLR MOSAIC
    printf 'line_%s_%s_%spx_%sbs_%se_fl%s_mo%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7"
}

weights_for() {  # weights_for FORM MODEL — le couple (poids, dataset) porte la formulation
    local form="$1" model="$2"
    case "$form" in
        det) echo "${MODELS_DIR}/${model}.pt" ;;
        seg) echo "${MODELS_DIR}/${model}-seg.pt" ;;
        *) echo "Formulation inconnue : $form" >&2; return 1 ;;
    esac
}

# Un YAML de dataset régénéré à chemin ABSOLU résolu sur la machine courante.
# Les config.yaml livrés avec les datasets ont porté pendant des mois un `path:`
# vers un dossier inexistant, et leur variante config_serveur.yaml en porte un
# autre, propre à un serveur donné. Reconstruire le YAML supprime la classe
# entière de ces pannes : rien à corriger à la main en changeant de machine.
dataset_yaml_for() {  # dataset_yaml_for FORM
    local form="$1" src dest
    case "$form" in
        det) src="${DATASET_ROOT}/${DET_DATASET}" ;;
        seg) src="${DATASET_ROOT}/${SEG_DATASET}" ;;
        *) echo "Formulation inconnue : $form" >&2; return 1 ;;
    esac
    [[ -d "$src" ]] || { echo "Dataset introuvable : $src" >&2; return 1; }
    local abs; abs=$(cd "$src" && pwd)
    dest="${GEN_DIR}/dataset_line_${form}.yaml"
    {
        echo "# Généré par scripts/train_line_sweep.sh — ne pas éditer."
        echo "# Chemin absolu résolu sur cette machine, pour ne pas dépendre du"
        echo "# dossier de travail ni du DATASETS_DIR d'ultralytics."
        echo "path: ${abs}"
        grep -v '^path:' "${src}/config.yaml" | grep -v '^#'
    } > "$dest"
    echo "$dest"
}

# ── Rendu des configs ─────────────────────────────────────────────────────────
# Les archétypes ne sont jamais modifiés : on en dérive une config par run, où
# seules les clés du protocole sont réécrites. Les substitutions conservent
# l'indentation d'origine (\1), donc restent valides si les archétypes sont
# reformatés.
render_train_config() {  # ... DEST NAME DATASET_YAML WEIGHTS IMGSZ BS EPOCHS FLIPLR MOSAIC
    local dest="$1" name="$2" dsyaml="$3" weights="$4" imgsz="$5" bs="$6" ep="$7" fl="$8" mo="$9"
    sed -e "s|^\( *\)run_name:.*|\1run_name: \"${name}\"|" \
        -e "s|^\( *\)train:.*|\1train: \"${dsyaml}\"|" \
        -e "s|^\( *\)pretrained_w:.*|\1pretrained_w: \"${weights}\"|" \
        -e "s|^\( *\)train_name:.*|\1train_name: \"${name}\"|" \
        -e "s|^\( *\)img_size:.*|\1img_size: ${imgsz}|" \
        -e "s|^\( *\)batch_size:.*|\1batch_size: ${bs}|" \
        -e "s|^\( *\)epochs:.*|\1epochs: ${ep}|" \
        -e "s|^\( *\)fliplr:.*|\1fliplr: ${fl}|" \
        -e "s|^\( *\)mosaic:.*|\1mosaic: ${mo}|" \
        -e "s|^\( *\)device:.*|\1device: \"${DEVICE}\"|" \
        -e "s|^\( *\)use_wandb:.*|\1use_wandb: ${USE_WANDB}|" \
        "$TEMPLATE_TRAIN" > "$dest"

    # Garde-fou : si l'archétype perd une clé, on s'en aperçoit ici plutôt qu'en
    # découvrant trois heures plus tard un run entraîné avec fliplr=0.5.
    local key
    for key in train_name fliplr mosaic pretrained_w; do
        grep -q "^ *${key}:" "$dest" \
            || { echo "  ✗ clé ${key} absente de ${TEMPLATE_TRAIN}" >&2; return 1; }
    done
    grep -q "fliplr: ${fl}\$" "$dest" \
        || { echo "  ✗ fliplr non substitué dans ${dest}" >&2; return 1; }
}

# img_size d'inférence = img_size d'entraînement, toujours. Le §1.5 mesure
# jusqu'à −41 % de map50-95 sur le seul désaccord entre les deux ; évaluer un
# modèle 1536 px à 640 px ne dirait rien de ce modèle.
render_eval_config() {  # ... DEST NAME MODEL_PATH IMGSZ
    local dest="$1" name="$2" model_path="$3" imgsz="$4"
    sed -e "s|^\( *\)run_name:.*|\1run_name: \"${name}\"|" \
        -e "s|^\( *\)model_path:.*|\1model_path: \"${model_path}\"|" \
        -e "s|^\( *\)img_size:.*|\1img_size: ${imgsz}|" \
        -e "s|^\( *\)device:.*|\1device: \"${EVAL_DEVICE}\"|" \
        -e "s|^\( *\)use_wandb:.*|\1use_wandb: ${USE_WANDB}|" \
        -e "s|^\( *\)save_image:.*|\1save_image: false|" \
        -e "s|^\( *\)test:.*|\1test: \"${DATA}\"|" \
        -e "s|^\( *\)restrict_to_layout:.*|\1restrict_to_layout: true|" \
        "$TEMPLATE_EVAL" > "$dest"

    grep -q 'restrict_to_layout: true' "$dest" \
        || { echo "  ✗ restrict_to_layout absent de ${TEMPLATE_EVAL}" >&2; return 1; }
    grep -q "\"${model_path}\"" "$dest" \
        || { echo "  ✗ model_path non substitué dans ${TEMPLATE_EVAL}" >&2; return 1; }
}

# ── Localisation des poids ────────────────────────────────────────────────────
# `project='LS-training'` étant relatif, ultralytics le résout sous son RUNS_DIR
# et la tâche : les poids atterrissent dans
#   runs/detect/LS-training/<name>/weights/best.pt   (détection)
#   runs/segment/LS-training/<name>/weights/best.pt  (segmentation)
# — pas dans ./LS-training/<name>/ comme le nom le suggère. Et un nom déjà pris
# est désambiguïsé par un suffixe numérique (<name>2, <name>3…). On cherche donc
# le best.pt le plus récent parmi les candidats plutôt que de supposer un chemin :
# sinon un run interrompu puis relancé fait évaluer les poids de la tentative
# précédente, et un changement de formulation fait chercher au mauvais endroit.
find_best_pt() {  # find_best_pt NAME
    local name="$1"
    find . -path "*/${TRAIN_PROJECT}/${name}/weights/best.pt" \
           -o -path "*/${TRAIN_PROJECT}/${name}[0-9]*/weights/best.pt" 2>/dev/null \
        | xargs -r ls -1t 2>/dev/null | head -1
}

# ── Exécution d'un run ───────────────────────────────────────────────────────
declare -a ROW_AXIS ROW_VALUE ROW_NAME
FAILED=""
FAILED_COUNT=0

# do_run AXE VALEUR FORM MODEL IMGSZ EPOCHS FLIPLR MOSAIC
# Entraîne si nécessaire, évalue si nécessaire, enregistre une ligne de tableau.
# Renvoie 0 si le run est exploitable (results.csv présent).
do_run() {
    local axis="$1" value="$2" form="$3" model="$4" imgsz="$5" ep="$6" fl="$7" mo="$8"
    local bs; bs=$(batch_for "$imgsz")
    local name; name=$(run_name_for "$form" "$model" "$imgsz" "$bs" "$ep" "$fl" "$mo")
    local weights; weights=$(weights_for "$form" "$model") || return 1
    local dsyaml; dsyaml=$(dataset_yaml_for "$form") || return 1
    local train_cfg="${GEN_DIR}/${name}_train.yml"
    local eval_cfg="${GEN_DIR}/${name}_eval.yml"
    local log="${LOG_DIR}/${name}.log"
    local result_csv="${OUTPUT_DIR}/${name}/line/results.csv"

    echo "--------------------------------------------------------"
    echo "  ${axis} = ${value}"
    echo "  ${name}"
    echo "    formulation=${form}  poids=${weights}"
    echo "    imgsz=${imgsz}  batch=${bs}  epochs=${ep}  fliplr=${fl}  mosaic=${mo}"

    if [[ ! -f "$weights" ]]; then
        echo "  ✗ poids de départ absents : ${weights}"
        FAILED+="   - ${name} (poids ${weights} absents)"$'\n'
        FAILED_COUNT=$((FAILED_COUNT+1))
        return 1
    fi

    render_train_config "$train_cfg" "$name" "$dsyaml" "$weights" \
                        "$imgsz" "$bs" "$ep" "$fl" "$mo" || return 1

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "    (dry-run) config d'entraînement : ${train_cfg}"
        ROW_AXIS+=("$axis"); ROW_VALUE+=("$value"); ROW_NAME+=("$name")
        return 1
    fi

    # ── entraînement ──
    local best; best=$(find_best_pt "$name")
    if [[ -n "$best" && "$RETRAIN" != "true" ]]; then
        echo "    [train] déjà entraîné → ${best} (réutilisé ; --retrain pour forcer)"
    else
        echo "    [train] → ${TRAIN_PROJECT}/${name}/"
        local t0; t0=$(date +%s)
        if ! docworkflow -c "$train_cfg" train -t line 2>&1 | tee -a "$log"; then
            echo "    ✗ l'entraînement a échoué (voir ${log})"
            FAILED+="   - ${name} (train)"$'\n'
            FAILED_COUNT=$((FAILED_COUNT+1))
            return 1
        fi
        local elapsed=$(( $(date +%s) - t0 ))
        # §4 du cahier des charges : le temps par époque dimensionne un futur
        # balayage plus large. C'est aussi la seule façon de voir qu'un run à
        # 1536 px coûte quatre fois un run à 640 px.
        printf "    ✓ entraîné en %dh%02dm (%.1f min/époque, device %s)\n" \
            $((elapsed/3600)) $(((elapsed%3600)/60)) \
            "$(awk -v e="$elapsed" -v n="$ep" 'BEGIN{print e/60/n}')" "$DEVICE"
        echo "${name},${form},${imgsz},${bs},${ep},${elapsed},${DEVICE}" >> "$TIMINGS"
        best=$(find_best_pt "$name")
    fi

    if [[ -z "$best" ]]; then
        echo "    ✗ aucun best.pt trouvé pour ${name}"
        FAILED+="   - ${name} (best.pt introuvable)"$'\n'
        FAILED_COUNT=$((FAILED_COUNT+1))
        return 1
    fi

    # ── predict + score sur le corpus gelé ──
    render_eval_config "$eval_cfg" "$name" "$best" "$imgsz" || return 1

    if [[ -f "$result_csv" && "$RESCORE" != "true" ]]; then
        echo "    [eval] déjà scoré → ${result_csv} (réutilisé ; --rescore pour forcer)"
    else
        echo "    [predict] → ${OUTPUT_DIR}/${name}/line (imgsz=${imgsz}, device ${EVAL_DEVICE})"
        if ! docworkflow -c "$eval_cfg" predict -t line 2>&1 | tee -a "$log"; then
            echo "    ✗ predict a échoué (voir ${log})"
            FAILED+="   - ${name} (predict)"$'\n'
            FAILED_COUNT=$((FAILED_COUNT+1))
            return 1
        fi
        echo "    [score]"
        if ! docworkflow -c "$eval_cfg" score -t line 2>&1 | tee -a "$log"; then
            echo "    ✗ score a échoué (voir ${log})"
            FAILED+="   - ${name} (score)"$'\n'
            FAILED_COUNT=$((FAILED_COUNT+1))
            return 1
        fi
    fi

    if [[ ! -f "$result_csv" ]]; then
        echo "    ✗ pas de results.csv pour ${name}"
        FAILED+="   - ${name} (pas de results.csv)"$'\n'
        FAILED_COUNT=$((FAILED_COUNT+1))
        return 1
    fi

    ROW_AXIS+=("$axis"); ROW_VALUE+=("$value"); ROW_NAME+=("$name")
    printf "    %s = %s\n" "$SELECT_METRIC" "$(fmt "$(csv_get "$result_csv" "$SELECT_METRIC")")"
    return 0
}

# ── Élection du gagnant d'un étage ───────────────────────────────────────────
# Sur SELECT_METRIC (map50-95 par défaut), mesurée sur le corpus de test gelé et
# non sur le split valid de CATMuS : c'est la seule mesure qui porte sur la cible.
# La sortie est le triplet "valeur<TAB>nom<TAB>métrique".
declare -a STAGE_VALUES STAGE_NAMES
elect() {
    local best_val="" best_name="" best_metric="" i m
    for i in "${!STAGE_NAMES[@]}"; do
        m=$(csv_get "${OUTPUT_DIR}/${STAGE_NAMES[$i]}/line/results.csv" "$SELECT_METRIC")
        [[ -n "$m" ]] || continue
        if [[ -z "$best_metric" ]] || awk -v a="$m" -v b="$best_metric" 'BEGIN{exit !(a>b)}'; then
            best_metric="$m"; best_val="${STAGE_VALUES[$i]}"; best_name="${STAGE_NAMES[$i]}"
        fi
    done
    [[ -n "$best_name" ]] || return 1
    printf '%s\t%s\t%s\n' "$best_val" "$best_name" "$best_metric"
}

stage_report() {  # stage_report TITRE AXE
    local title="$1" axis="$2" i m
    echo ""
    echo "  ── ${title} : classement sur ${SELECT_METRIC} ──"
    printf "     %-14s %-10s %s\n" "$axis" "$SELECT_METRIC" "run"
    for i in "${!STAGE_NAMES[@]}"; do
        m=$(csv_get "${OUTPUT_DIR}/${STAGE_NAMES[$i]}/line/results.csv" "$SELECT_METRIC")
        printf "     %-14s %-10s %s\n" "${STAGE_VALUES[$i]}" "$(fmt "$m")" "${STAGE_NAMES[$i]}"
    done
}

# ── Préflight ─────────────────────────────────────────────────────────────────
# Un balayage complet dure des heures : mieux vaut échouer ici que découvrir au
# dixième run qu'un dataset ou une dépendance manque.
PIXI_ENV="${PIXI_ENVIRONMENT_NAME:-?}"
errors=0

for f in "$TEMPLATE_TRAIN" "$TEMPLATE_EVAL"; do
    [[ -f "$f" ]] || { echo "✗ archétype manquant : $f" >&2; errors=$((errors+1)); }
done
[[ -d "$DATA" ]] || { echo "✗ corpus de test introuvable : $DATA" >&2; errors=$((errors+1)); }

for ds in "$DET_DATASET" "$SEG_DATASET"; do
    [[ -f "${DATASET_ROOT}/${ds}/config.yaml" ]] \
        || { echo "✗ dataset introuvable : ${DATASET_ROOT}/${ds}/config.yaml" >&2
             echo "  (ajuster --dataset-root)" >&2; errors=$((errors+1)); }
done

command -v docworkflow >/dev/null \
    || { echo "✗ docworkflow introuvable — lancer via 'pixi run -e inference ...'" >&2
         errors=$((errors+1)); }

python -c "import ultralytics" >/dev/null 2>&1 \
    || { echo "✗ ultralytics absent de l'environnement pixi '${PIXI_ENV}'." >&2
         echo "  L'entraînement YOLO exige 'inference' (l'env 'train' ne l'a pas)." >&2
         errors=$((errors+1)); }

if [[ "$DEVICE" != "cpu" ]]; then
    if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "✗ --device ${DEVICE} demandé mais torch.cuda.is_available() est faux" >&2
        echo "  dans l'environnement pixi '${PIXI_ENV}'. Utiliser 'inference' sur une" >&2
        echo "  machine GPU, ou --device cpu (un entraînement CPU est hors de portée" >&2
        echo "  en pratique : compter des jours par run)." >&2
        errors=$((errors+1))
    else
        echo "  GPU : $(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)"
    fi
fi

if [[ "$EVAL_DEVICE" != "cpu" ]]; then
    python -c "
import sys, torch
try:
    torch.zeros(1).to('${EVAL_DEVICE}')
except Exception as e:
    print(f'✗ --eval-device ${EVAL_DEVICE} inutilisable : {e}', file=sys.stderr)
    sys.exit(1)
" || errors=$((errors+1))
fi

[[ $errors -eq 0 ]] || { echo "" >&2; echo "Préflight en échec, rien n'a été lancé." >&2; exit 1; }

echo "========================================================"
echo " Balayage d'entraînement — segmentation de ligne"
echo "   étages            : ${STAGES}"
echo "   datasets          : ${DATASET_ROOT}/{${DET_DATASET},${SEG_DATASET}}"
echo "   corpus de test    : ${DATA} ($(find "$DATA" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l) pages)"
echo "   device train/eval : ${DEVICE} / ${EVAL_DEVICE}"
echo "   batch             : ${BATCH_SIZE}${BATCH_MAP:+ (carte : ${BATCH_MAP})}"
echo "   graine            : ${SEED}"
echo "   départage sur     : ${SELECT_METRIC} (corpus gelé, pas le valid CATMuS)"
echo "   sorties           : ${SWEEP_DIR}"
echo "========================================================"

[[ -f "$TIMINGS" ]] || echo "name,formulation,imgsz,batch,epochs,seconds,device" > "$TIMINGS"

CUR_FORM="det"

for stage in $STAGES; do
    STAGE_VALUES=(); STAGE_NAMES=()
    echo ""
    echo "########################################################"

    case "$stage" in
    0)
        echo "# Étage 0 — référence corrigée"
        echo "# Remplace yolo26s_line_640px_16bs_50e.pt, entraîné avec les"
        echo "# augmentations par défaut (fliplr=0.5, mosaic=1.0)."
        echo "########################################################"
        do_run "reference" "yolo26s/det/640/${BASE_EPOCHS}e" \
               det yolo26s 640 "$BASE_EPOCHS" 0.0 0.0
        ;;
    1)
        echo "# Étage 1 — résolution d'entraînement (détection fixée)"
        echo "# Ablation prioritaire du §1.5 : jamais faite à l'entraînement."
        echo "########################################################"
        if [[ "$FIXED_IMGSZ" == "true" ]]; then
            echo "  (img_size imposé à ${CUR_IMGSZ} par --fix-img-size, étage sauté)"
        else
            for imgsz in 640 1280 1536; do
                do_run "img_size" "$imgsz" det "$CUR_MODEL" "$imgsz" "$CUR_EPOCHS" \
                       "$CUR_FLIPLR" "$CUR_MOSAIC" \
                    && { STAGE_VALUES+=("$imgsz"); STAGE_NAMES+=("${ROW_NAME[-1]}"); }
            done
            stage_report "Étage 1 (résolution)" "img_size"
            if won=$(elect); then
                CUR_IMGSZ=$(cut -f1 <<<"$won")
                echo "  → gagnant : img_size=${CUR_IMGSZ} ($(fmt "$(cut -f3 <<<"$won")"))"
            else
                echo "  ⚠ aucun run exploitable : img_size reste à ${CUR_IMGSZ}"
            fi
        fi
        ;;
    2)
        echo "# Étage 2 — augmentations, à img_size=${CUR_IMGSZ}"
        echo "# Grille croisée assumée : fliplr et mosaic sont les deux"
        echo "# paramètres déjà identifiés comme suspects (§3.1)."
        echo "########################################################"
        if [[ "$FIXED_FLIPLR" == "true" && "$FIXED_MOSAIC" == "true" ]]; then
            echo "  (fliplr/mosaic imposés par --fix-*, étage sauté)"
        else
            for fl in 0.0 0.5; do
                for mo in 0.0 0.2 1.0; do
                    do_run "fliplr×mosaic" "fl=${fl} mo=${mo}" det "$CUR_MODEL" \
                           "$CUR_IMGSZ" "$CUR_EPOCHS" "$fl" "$mo" \
                        && { STAGE_VALUES+=("fl${fl}/mo${mo}"); STAGE_NAMES+=("${ROW_NAME[-1]}"); }
                done
            done
            stage_report "Étage 2 (augmentations)" "fliplr/mosaic"
            if won=$(elect); then
                combo=$(cut -f1 <<<"$won")
                CUR_FLIPLR="${combo%%/*}"; CUR_FLIPLR="${CUR_FLIPLR#fl}"
                CUR_MOSAIC="${combo##*/mo}"
                echo "  → gagnant : fliplr=${CUR_FLIPLR} mosaic=${CUR_MOSAIC} ($(fmt "$(cut -f3 <<<"$won")"))"
            else
                echo "  ⚠ aucun run exploitable : fliplr=${CUR_FLIPLR} mosaic=${CUR_MOSAIC} conservés"
            fi
        fi
        ;;
    3)
        echo "# Étage 3 — taille de modèle, réglages 1-2 fixés"
        echo "# yolo26n et yolo26s sont les seuls backbones ligne présents dans"
        echo "# ${MODELS_DIR}. Tester plus gros suppose de télécharger"
        echo "# yolo26m/yolo26l — à signaler explicitement si c'est fait."
        echo "########################################################"
        if [[ "$FIXED_MODEL" == "true" ]]; then
            echo "  (modèle imposé à ${CUR_MODEL} par --fix-model, étage sauté)"
        else
            for m in yolo26n yolo26s; do
                do_run "model" "$m" det "$m" "$CUR_IMGSZ" "$CUR_EPOCHS" \
                       "$CUR_FLIPLR" "$CUR_MOSAIC" \
                    && { STAGE_VALUES+=("$m"); STAGE_NAMES+=("${ROW_NAME[-1]}"); }
            done
            stage_report "Étage 3 (taille de modèle)" "model"
            if won=$(elect); then
                CUR_MODEL=$(cut -f1 <<<"$won")
                echo "  → gagnant : ${CUR_MODEL} ($(fmt "$(cut -f3 <<<"$won")"))"
            else
                echo "  ⚠ aucun run exploitable : ${CUR_MODEL} conservé"
            fi
        fi
        ;;
    4)
        echo "# Étage 4 — nombre d'époques, réglages 1-3 fixés"
        echo "# Rien en dessous de 30 : le seul yolo26n du benchmark actuel était"
        echo "# entraîné à 10 époques et n'est pas un point de comparaison (§3.1)."
        echo "########################################################"
        if [[ "$FIXED_EPOCHS" == "true" ]]; then
            echo "  (époques imposées à ${CUR_EPOCHS} par --fix-epochs, étage sauté)"
        else
            for ep in 30 50 100; do
                do_run "epochs" "$ep" det "$CUR_MODEL" "$CUR_IMGSZ" "$ep" \
                       "$CUR_FLIPLR" "$CUR_MOSAIC" \
                    && { STAGE_VALUES+=("$ep"); STAGE_NAMES+=("${ROW_NAME[-1]}"); }
            done
            stage_report "Étage 4 (époques)" "epochs"
            if won=$(elect); then
                CUR_EPOCHS=$(cut -f1 <<<"$won")
                echo "  → gagnant : ${CUR_EPOCHS} époques ($(fmt "$(cut -f3 <<<"$won")"))"
            else
                echo "  ⚠ aucun run exploitable : ${CUR_EPOCHS} époques conservées"
            fi
        fi
        ;;
    5)
        echo "# Étage 5 — formulation : détection vs segmentation"
        echo "# La comparaison contrôlée qui manque au plan (§2.1). MÊME config"
        echo "# que le gagnant détection, seul le couple (poids, dataset) change."
        echo "# Attention à la lecture : zonemap/score ne compare valablement"
        echo "# qu'au sein d'une même formulation (§1.4) — trancher sur mAP et"
        echo "# P/R/F1, plus une lecture visuelle."
        echo "# L'OBB rejoindra cette comparaison dans un second temps."
        echo "########################################################"
        for form in det seg; do
            do_run "formulation" "$form" "$form" "$CUR_MODEL" "$CUR_IMGSZ" \
                   "$CUR_EPOCHS" "$CUR_FLIPLR" "$CUR_MOSAIC" \
                && { STAGE_VALUES+=("$form"); STAGE_NAMES+=("${ROW_NAME[-1]}"); }
        done
        stage_report "Étage 5 (formulation)" "formulation"
        if won=$(elect); then
            CUR_FORM=$(cut -f1 <<<"$won")
            echo "  → meilleure sur ${SELECT_METRIC} : ${CUR_FORM} ($(fmt "$(cut -f3 <<<"$won")"))"
            echo "    (à confirmer visuellement avant d'en faire une recommandation)"
        fi
        ;;
    *)
        echo "# Étage inconnu : ${stage} — ignoré"
        ;;
    esac
done

# ── Tableau récapitulatif ─────────────────────────────────────────────────────
# Une ligne par run, l'axe varié en première colonne, pour la lecture « avant /
# après » du §1.2 du plan. Les métriques principales viennent de results.csv, la
# décomposition ZoneMapAlt de results_zonemap.csv.
{
    echo "axe,valeur,run,formulation,imgsz,batch,epochs,min_par_epoque,map50,map75,map50-95,P,R,F1,zm_score,zm_match,zm_miss,zm_false_alarm,zm_n_miss,zm_n_false_alarm,zm_n_split,zm_n_merge,zm_n_multiple"
    for i in "${!ROW_NAME[@]}"; do
        n="${ROW_NAME[$i]}"
        rc="${OUTPUT_DIR}/${n}/line/results.csv"
        zc="${OUTPUT_DIR}/${n}/line/results_zonemap.csv"
        [[ -f "$rc" ]] || continue
        # line_<form>_<model>_<imgsz>px_<bs>bs_<ep>e_fl<fl>_mo<mo> : le nom porte
        # tous les hyperparamètres, il est donc la source la plus fiable — un run
        # réutilisé d'un balayage antérieur n'a pas forcément d'entrée de durée.
        read -r tform timgsz tbatch tep < <(sed -E \
            's|^line_([a-z]+)_[^_]+_([0-9]+)px_([0-9]+)bs_([0-9]+)e_.*|\1 \2 \3 \4|' <<<"$n")
        tsec=$(awk -F',' -v n="$n" '$1==n{print $6; exit}' "$TIMINGS")
        mpe="-"
        [[ -n "${tsec:-}" && -n "${tep:-}" ]] \
            && mpe=$(awk -v s="$tsec" -v e="$tep" 'BEGIN{printf "%.2f", s/60/e}')
        printf '%s,%s,%s,%s,%s,%s,%s,%s' \
            "${ROW_AXIS[$i]}" "\"${ROW_VALUE[$i]}\"" "$n" \
            "${tform:-}" "${timgsz:-}" "${tbatch:-}" "${tep:-}" "${mpe}"
        for col in dataset_test/map50 dataset_test/map75 dataset_test/map50-95 \
                   dataset_test/precision dataset_test/recall dataset_test/f1; do
            printf ',%s' "$(csv_get "$rc" "$col")"
        done
        for key in zonemap/score zonemap/match zonemap/miss zonemap/false_alarm \
                   zonemap/n_miss zonemap/n_false_alarm zonemap/n_split \
                   zonemap/n_merge zonemap/n_multiple; do
            printf ',%s' "$(kv_get "$zc" "$key")"
        done
        echo ""
    done
} > "$SUMMARY"

echo ""
echo "========================================================"
echo " Configuration retenue au terme du balayage"
echo "   formulation : ${CUR_FORM}"
echo "   modèle      : ${CUR_MODEL}"
echo "   img_size    : ${CUR_IMGSZ} (entraînement ET inférence)"
echo "   fliplr      : ${CUR_FLIPLR}"
echo "   mosaic      : ${CUR_MOSAIC}"
echo "   époques     : ${CUR_EPOCHS}"
echo "   batch       : $(batch_for "$CUR_IMGSZ")"
echo ""
echo " Reprendre le balayage à un étage ultérieur sans rejouer les précédents :"
echo "   ./scripts/train_line_sweep.sh --stages \"5\" \\"
echo "       --fix-img-size ${CUR_IMGSZ} --fix-fliplr ${CUR_FLIPLR} \\"
echo "       --fix-mosaic ${CUR_MOSAIC} --fix-model ${CUR_MODEL} --fix-epochs ${CUR_EPOCHS}"
echo "========================================================"
if [[ -s "$SUMMARY" ]]; then
    echo " Récapitulatif : ${SUMMARY}"
    command -v column >/dev/null && column -s, -t "$SUMMARY" || cat "$SUMMARY"
fi
echo " Durées        : ${TIMINGS}"
echo " Journaux      : ${LOG_DIR}"
if [[ "$USE_WANDB" == "true" ]]; then
    echo " wandb         : entraînements sous le projet LS-training,"
    echo "                 predict/score sous LS-comparison — deux séries distinctes."
fi

if [[ $FAILED_COUNT -gt 0 ]]; then
    echo ""
    echo " ${FAILED_COUNT} run(s) en échec :"
    printf '%s' "$FAILED"
    exit 1
fi
