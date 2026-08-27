#!/bin/bash
#
# benchmark_line_models.sh — predict + score de tous les modèles de segmentation
# de ligne de src/tasks/line/models sur un jeu de test, avec trace wandb.
#
# Un modèle .mlmodel est passé à l'archétype Kraken, un modèle .pt à l'archétype
# YOLO. Les archétypes (configs/line/*.yml) ne sont jamais modifiés : le script
# en dérive une config par run dans configs/line/generated/, où seuls run_name,
# model_path, img_size, device, save_image, data.test et restrict_to_layout
# sont réécrits.
#
# Protocole (cf. 02_analyse_et_plan_action.md) :
#   - restrict_to_layout=true pour TOUS les candidats. Kraken reçoit les zones en
#     entrée d'inférence et ne peut structurellement pas prédire hors zone ; YOLO
#     voit la page entière. Sans cette option, les lignes YOLO hors zone
#     survivent en pseudo-zones sans TAGREFS et sont comptées en faux positifs
#     alors que les lignes de référence correspondantes sont filtrées (§1.4).
#   - §2.5 « Résolution » : chaque modèle YOLO est rejoué à plusieurs img_size.
#     Attention à la lecture — mesuré le 27/08, ce balayage donne la sensibilité
#     à la résolution *d'inférence*, à modèle figé : des modèles entraînés à 640
#     y perdent jusqu'à 41 % de map50-95 à 1536. Il ne teste PAS l'hypothèse du
#     §2.5, qui porte sur la résolution d'entraînement — celle-là ne se teste
#     qu'en réentraînant. Kraken n'a pas de paramètre équivalent, joué une fois.
#
# Usage :
#   ./scripts/benchmark_line_models.sh [options]
#
#   --data PATH        jeu de test (défaut : ../data/ICDAR_CMMHWR_original_data)
#   --device DEV       cpu | cuda (défaut : cpu). cuda exige l'environnement
#                      pixi `inference` : `main` embarque un pytorch CPU only.
#                        pixi run -e inference ./scripts/benchmark_line_models.sh --device cuda
#   --batch-size N     batch YOLO (défaut : valeur de l'archétype)
#   --img-sizes "..."  tailles YOLO à balayer (défaut : "640 1280 1536")
#   --models "..."     motifs à retenir, séparés par des espaces (défaut : tous)
#   --restrict-to-layout true|false   (défaut : true — voir ci-dessus)
#   --save-images      recopie les images à côté des ALTO prédits (défaut : non ;
#                      ~1,5 Go par run sur ce jeu, seulement utile à
#                      `docworkflow print -t line`)
#   --no-wandb         désactive la trace wandb
#   --skip-predict     ne fait que (re)scorer des prédictions déjà produites
#   --dry-run          affiche les runs et les configs générées, n'exécute rien
#
set -uo pipefail

# Ctrl+C tue le `docworkflow` en cours mais rend la main à la boucle, qui
# enchaînerait sur le run suivant : il faut sortir explicitement du script.
on_interrupt() {
    echo ""
    echo "Interrompu — les prédictions déjà écrites sont conservées ;"
    echo "relancer la même commande reprend là où elle s'est arrêtée."
    exit 130
}
trap on_interrupt INT TERM

cd "$(dirname "$0")/.."

MODELS_DIR="src/tasks/line/models"
TEMPLATE_KRAKEN="configs/line/kraken_line.yml"
TEMPLATE_YOLO="configs/line/yolo_line.yml"
GEN_DIR="configs/line/generated"

DATA="../data/ICDAR_CMMHWR_original_data"
DEVICE="cpu"
IMG_SIZES="640"
BATCH_SIZE=""
MODEL_FILTERS=""
RESTRICT_TO_LAYOUT="true"
SAVE_IMAGE="false"
USE_WANDB="true"
SKIP_PREDICT="false"
DRY_RUN="false"

# Poids ultralytics pré-entraînés COCO (yolo26n.pt, yolo26s-seg.pt, ...) : ce
# sont les points de départ d'entraînement, pas des modèles de ligne. Ils
# détecteraient des chiens et des voitures. On les écarte de la comparaison.
is_pretrained_backbone() {
    [[ "$1" =~ ^yolo(v)?[0-9]+[nsmlx](-seg|-obb|-pose|-cls)?$ ]]
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)        DATA="$2"; shift 2 ;;
        --device)      DEVICE="$2"; shift 2 ;;
        --img-sizes)   IMG_SIZES="$2"; shift 2 ;;
        --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
        --models)      MODEL_FILTERS="$2"; shift 2 ;;
        --restrict-to-layout) RESTRICT_TO_LAYOUT="$2"; shift 2 ;;
        --save-images) SAVE_IMAGE="true"; shift ;;
        --no-wandb)    USE_WANDB="false"; shift ;;
        --skip-predict) SKIP_PREDICT="true"; shift ;;
        --dry-run)     DRY_RUN="true"; shift ;;
        -h|--help)     sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "Option inconnue : $1" >&2; exit 2 ;;
    esac
done

for f in "$TEMPLATE_KRAKEN" "$TEMPLATE_YOLO"; do
    [[ -f "$f" ]] || { echo "Archétype manquant : $f" >&2; exit 1; }
done
[[ -d "$DATA" ]] || { echo "Jeu de test introuvable : $DATA" >&2; exit 1; }

OUTPUT_DIR=$(grep -m1 '^output_dir:' "$TEMPLATE_YOLO" | sed 's/output_dir: *"\?\([^"]*\)"\?.*/\1/')
OUTPUT_DIR="${OUTPUT_DIR:-results}"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_DIR="${OUTPUT_DIR}/line_benchmark_${STAMP}"
LOG_DIR="${SUMMARY_DIR}/logs"

mkdir -p "$GEN_DIR" "$LOG_DIR"

# Écrit une config dérivée de $1 dans $2, en réécrivant les clés du protocole.
# Les substitutions conservent l'indentation d'origine (\1) : elles sont donc
# insensibles à la mise en forme des archétypes.
render_config() {
    local template="$1" dest="$2" run_name="$3" model_path="$4" img_size="$5"
    sed -e "s|^\( *\)run_name:.*|\1run_name: \"${run_name}\"|" \
        -e "s|^\( *\)model_path:.*|\1model_path: \"${model_path}\"|" \
        -e "s|^\( *\)device:.*|\1device: \"${DEVICE}\"|" \
        -e "s|^\( *\)use_wandb:.*|\1use_wandb: ${USE_WANDB}|" \
        -e "s|^\( *\)save_image:.*|\1save_image: ${SAVE_IMAGE}|" \
        -e "s|^\( *\)test:.*|\1test: \"${DATA}\"|" \
        -e "s|^\( *\)restrict_to_layout:.*|\1restrict_to_layout: ${RESTRICT_TO_LAYOUT}|" \
        -e "s|^\( *\)img_size:.*|\1img_size: ${img_size}|" \
        "$template" > "$dest"

    if [[ -n "$BATCH_SIZE" ]]; then
        sed -i -e "s|^\( *\)batch_size:.*|\1batch_size: ${BATCH_SIZE}|" "$dest"
    fi

    # Garde-fou : si l'archétype perd une clé du protocole, on s'en aperçoit ici
    # plutôt qu'en lisant des chiffres non comparables trois heures plus tard.
    grep -q 'restrict_to_layout:' "$dest" \
        || { echo "  ✗ restrict_to_layout absent de ${template}" >&2; return 1; }
    grep -q "\"${model_path}\"" "$dest" \
        || { echo "  ✗ model_path non substitué dans ${template}" >&2; return 1; }
}

# --- Inventaire des runs ------------------------------------------------------
declare -a RUN_NAMES RUN_CONFIGS RUN_MODELS
matches_filter() {
    [[ -z "$MODEL_FILTERS" ]] && return 0
    local name="$1" pat
    for pat in $MODEL_FILTERS; do
        [[ "$name" == *"$pat"* ]] && return 0
    done
    return 1
}

shopt -s nullglob
for model in "$MODELS_DIR"/*.mlmodel "$MODELS_DIR"/*.pt; do
    stem=$(basename "$model"); stem="${stem%.*}"

    if is_pretrained_backbone "$stem"; then
        echo "  (ignoré, backbone pré-entraîné COCO) ${stem}"
        continue
    fi
    matches_filter "$stem" || continue

    case "$model" in
        *.mlmodel)
            run_name="line_kraken_${stem}"
            cfg="${GEN_DIR}/${run_name}.yml"
            render_config "$TEMPLATE_KRAKEN" "$cfg" "$run_name" "$model" "" || exit 1
            RUN_NAMES+=("$run_name"); RUN_CONFIGS+=("$cfg"); RUN_MODELS+=("$stem")
            ;;
        *.pt)
            for imgsz in $IMG_SIZES; do
                run_name="line_yolo_${stem}_imgsz${imgsz}"
                cfg="${GEN_DIR}/${run_name}.yml"
                render_config "$TEMPLATE_YOLO" "$cfg" "$run_name" "$model" "$imgsz" || exit 1
                RUN_NAMES+=("$run_name"); RUN_CONFIGS+=("$cfg"); RUN_MODELS+=("$stem")
            done
            ;;
    esac
done
shopt -u nullglob

TOTAL=${#RUN_NAMES[@]}
[[ $TOTAL -gt 0 ]] || { echo "Aucun modèle à évaluer dans ${MODELS_DIR}." >&2; exit 1; }

# --- Préflight ----------------------------------------------------------------
# Un balayage complet dure des heures : mieux vaut échouer ici que découvrir au
# vingtième traceback qu'une dépendance manque. Les vérifications portent sur
# l'interpréteur de l'environnement pixi courant, celui qui exécutera les runs.
PIXI_ENV="${PIXI_ENVIRONMENT_NAME:-?}"
has_module() { python -c "import $1" >/dev/null 2>&1; }
preflight_errors=0

command -v docworkflow >/dev/null || {
    echo "✗ docworkflow introuvable — lancer le script via 'pixi run -e main ...'" >&2
    preflight_errors=$((preflight_errors+1))
}

n_kraken=0; n_yolo=0
for name in "${RUN_NAMES[@]}"; do
    case "$name" in line_kraken_*) n_kraken=$((n_kraken+1)) ;; line_yolo_*) n_yolo=$((n_yolo+1)) ;; esac
done

if [[ $n_kraken -gt 0 ]] && ! has_module yaltai; then
    # kraken_line.py importe yaltai.models.krakn au chargement du module : sans
    # yaltai, tous les runs Kraken échouent à l'import, avant toute inférence.
    echo "✗ module 'yaltai' absent de l'environnement pixi '${PIXI_ENV}'," >&2
    echo "  or les ${n_kraken} run(s) Kraken en dépendent. Installer avec :" >&2
    echo "      pixi run -e ${PIXI_ENV} install-yaltai" >&2
    echo "  (ou relancer avec --models pour n'évaluer que les modèles YOLO)" >&2
    preflight_errors=$((preflight_errors+1))
fi

if [[ $n_yolo -gt 0 ]] && ! has_module ultralytics; then
    echo "✗ module 'ultralytics' absent de l'environnement pixi '${PIXI_ENV}'." >&2
    preflight_errors=$((preflight_errors+1))
fi

if [[ "$DEVICE" == cuda* ]]; then
    if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "✗ --device ${DEVICE} demandé mais torch.cuda.is_available() est faux" >&2
        echo "  dans l'environnement pixi '${PIXI_ENV}'. L'environnement 'main'" >&2
        echo "  embarque un pytorch CPU only ; utiliser 'inference' :" >&2
        echo "      pixi run -e inference ./scripts/benchmark_line_models.sh --device cuda" >&2
        preflight_errors=$((preflight_errors+1))
    else
        gpu=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        echo "  GPU détecté : ${gpu}"
    fi
fi

[[ $preflight_errors -eq 0 ]] || { echo "" >&2; echo "Préflight en échec, rien n'a été lancé." >&2; exit 1; }

echo "========================================================"
echo " Benchmark segmentation de ligne"
echo "   jeu de test        : ${DATA}"
echo "   device             : ${DEVICE}"
echo "   restrict_to_layout : ${RESTRICT_TO_LAYOUT}"
echo "   img_size (YOLO)    : ${IMG_SIZES}"
echo "   runs               : ${TOTAL}"
echo "   sorties            : ${SUMMARY_DIR}"
echo "========================================================"
for i in "${!RUN_NAMES[@]}"; do echo "  [$((i+1))/${TOTAL}] ${RUN_NAMES[$i]}"; done
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "--dry-run : configs générées dans ${GEN_DIR}, rien n'a été exécuté."
    exit 0
fi

# --- Exécution ----------------------------------------------------------------
# Pas de `set -e` : un modèle qui casse ne doit pas emporter le balayage.
FAILED=""
FAILED_COUNT=0
for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"
    cfg="${RUN_CONFIGS[$i]}"
    result_dir="${OUTPUT_DIR}/${run_name}/line"
    log="${LOG_DIR}/${run_name}.log"

    echo "--------------------------------------------------------"
    echo "[$((i+1))/${TOTAL}] ${run_name}"
    echo "--------------------------------------------------------"
    started=$(date +%s)

    if [[ "$SKIP_PREDICT" != "true" ]]; then
        echo "  [predict] → ${result_dir}"
        if ! docworkflow -c "$cfg" predict -t line 2>&1 | tee -a "$log"; then
            echo "  ✗ predict a échoué (voir ${log})"
            FAILED+="   - ${run_name} (predict)"$'\n'; FAILED_COUNT=$((FAILED_COUNT+1))
            continue
        fi
    fi

    if [[ ! -d "$result_dir" ]]; then
        echo "  ✗ aucune prédiction dans ${result_dir}"
        FAILED+="   - ${run_name} (pas de prédictions)"$'\n'; FAILED_COUNT=$((FAILED_COUNT+1))
        continue
    fi

    echo "  [score]"
    if ! docworkflow -c "$cfg" score -t line 2>&1 | tee -a "$log"; then
        echo "  ✗ score a échoué (voir ${log})"
        FAILED+="   - ${run_name} (score)"$'\n'; FAILED_COUNT=$((FAILED_COUNT+1))
        continue
    fi

    elapsed=$(( $(date +%s) - started ))
    pages=$(find "$DATA" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
    # §2.2 « Pages/h » : à consigner systématiquement, c'est le chiffre qui
    # dimensionne l'objectif 2 (plusieurs centaines de milliers de pages).
    if [[ "$SKIP_PREDICT" != "true" && $elapsed -gt 0 ]]; then
        printf "  ✓ %s en %dm%02ds (%.0f pages/h, %s)\n" \
            "$run_name" $((elapsed/60)) $((elapsed%60)) \
            "$(awk -v p="$pages" -v e="$elapsed" 'BEGIN{print p*3600/e}')" "$DEVICE"
        echo "${run_name},${pages},${elapsed},${DEVICE}" >> "${SUMMARY_DIR}/timings.csv"
    else
        printf "  ✓ %s en %dm%02ds\n" "$run_name" $((elapsed/60)) $((elapsed%60))
    fi
    echo ""
done

# --- Tableau récapitulatif -----------------------------------------------------
# Fusionne les results.csv de chaque run. L'appariement se fait **par nom de
# colonne** : recopier la ligne de valeurs sous l'en-tête du premier fichier
# suffirait tant que tous les runs produisent exactement les mêmes colonnes dans
# le même ordre, hypothèse qu'un modèle en échec partiel ou un jeu de métriques
# différent suffit à briser — et le tableau devient alors silencieusement faux.
SUMMARY="${SUMMARY_DIR}/summary.csv"
declare -a COLUMNS=()
declare -A SEEN=()
declare -a ROW_KEYS=()

for i in "${!RUN_NAMES[@]}"; do
    csv="${OUTPUT_DIR}/${RUN_NAMES[$i]}/line/results.csv"
    [[ -f "$csv" ]] || continue
    IFS=',' read -r -a hdr < <(head -1 "$csv")
    for col in "${hdr[@]}"; do
        if [[ -z "${SEEN[$col]:-}" ]]; then SEEN[$col]=1; COLUMNS+=("$col"); fi
    done
    ROW_KEYS+=("$i")
done

if [[ ${#ROW_KEYS[@]} -gt 0 ]]; then
    { IFS=','; printf 'run_name,model,%s\n' "${COLUMNS[*]}"; } > "$SUMMARY"
    for i in "${ROW_KEYS[@]}"; do
        csv="${OUTPUT_DIR}/${RUN_NAMES[$i]}/line/results.csv"
        IFS=',' read -r -a hdr < <(head -1 "$csv")
        IFS=',' read -r -a val < <(sed -n '2p' "$csv")
        declare -A cell=()
        for j in "${!hdr[@]}"; do cell["${hdr[$j]}"]="${val[$j]:-}"; done
        line="${RUN_NAMES[$i]},${RUN_MODELS[$i]}"
        for col in "${COLUMNS[@]}"; do line+=",${cell[$col]:-}"; done
        echo "$line" >> "$SUMMARY"
        unset cell
    done
fi

echo "========================================================"
if [[ -f "$SUMMARY" ]]; then
    echo " Récapitulatif : ${SUMMARY}"
    command -v column >/dev/null && column -s, -t "$SUMMARY" || cat "$SUMMARY"
else
    echo " Aucun results.csv produit."
fi
[[ -f "${SUMMARY_DIR}/timings.csv" ]] && echo " Débits        : ${SUMMARY_DIR}/timings.csv"
echo " Journaux      : ${LOG_DIR}"
[[ "$USE_WANDB" == "true" ]] && echo " wandb         : projet LS-comparison, un run par ligne du tableau"

if [[ $FAILED_COUNT -gt 0 ]]; then
    echo ""
    echo " ${FAILED_COUNT} run(s) en échec :"
    printf '%s' "$FAILED"
    exit 1
fi
echo " ${TOTAL} run(s) terminés."
