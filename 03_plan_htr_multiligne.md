# Plan d'action — HTR multiligne (`VLMMultiLineHTR`)

> Document opérationnel, état au 13 octobre 2026. Porte uniquement sur le §4.1 du document 2
> (HTR multiligne). Le contexte à l'échelle du document (§4.2 : glossaire, métadonnées,
> soft prompt) est hors périmètre et fera l'objet d'un document 4, conditionné aux résultats
> obtenus ici. Correspond aux items 3 et 4 de la Phase B du document 2.
>
> Rappel d'environnement : pas de GPU local. Toutes les étapes de code et de petits tests sont
> à préparer et valider en local (CPU, fixtures synthétiques) ; tout entraînement ou inférence
> en volume s'exécute après un `git push` / `git pull` sur le serveur du laboratoire, ou sur
> Jean Zay pour les runs de la taille de ceux de MEDUSA (cf. `configs/hidden/jeanzay/`).

---

## 0. Décisions de conception verrouillées

| # | Question | Décision | Source |
|---|----------|----------|--------|
| 1 | Constitution du crop | **Rectangle englobant les *k* lignes** (pas de composite empilé). Robuste à un layout imprécis et aux doublons ; accepte le bruit inter-lignes en échange | message utilisateur |
| 2 | Masquage | Pas de masque polygonal sur l'union : on garde le rectangle brut, le bruit inter-lignes (marges, amorces de lignes voisines) est accepté par construction (cf. décision 1) | message utilisateur |
| 3 | Format de sortie | Une ligne de texte par ligne d'image, comme MEDUSA ligne-à-ligne classique ; sortie ALTO identique. Séparateur = retour à la ligne (`\n`) dans un premier temps — le jeton dédié proposé au doc 2 est gardé en réserve si le `\n` s'avère ambigu en pratique (§3, Phase 0.3) | message utilisateur + arbitrage ci-dessous |
| 4 | Budget de pixels | Doit s'adapter à *k*, mais l'arbitrage contexte/résolution est **mesuré, pas supposé** (§3, Phase 3.3) | doc 2 + message utilisateur |
| 5 | Fenêtre roulante | Agrégation par vote (ROVER), en option activable/désactivable ; comparer avec et sans. Le désaccord entre fenêtres est regardé comme signal de fiabilité | message utilisateur |
| 6 | Balayage de *k* | *k* = 1, 2, 4, 8 (1 = contrôle, c'est le Medusa ligne-à-ligne déjà mesuré) | doc 2 + message utilisateur |
| 7 | Dédoublonnage | Un test de détection de lignes de sortie dupliquées (répétition dégénérée), appliqué en post-traitement, avec garde-fou pour ne pas collapser une répétition réelle du texte source | message utilisateur, précisé §3 Phase 0.3 |
| 8 | Entrée / cible | Pipeline de bout en bout, polygones de ligne en entraînement, baseline/polygone en sortie — identique à MEDUSA ligne-à-ligne (décisions 2-3 de l'annexe du doc 1) | déjà tranché |

**Séquence macro décidée par l'utilisateur**, qui structure les phases ci-dessous :

1. **Zero-shot** : observer le comportement de MEDUSA tel quel sur des crops multiligne, sans réentraînement.
2. **Finetune MEDUSA** (poursuite du LoRA existant) sur du multiligne.
3. **Réentraînement complet** depuis le même Qwen3.5 pré-entraîné que MEDUSA, avec balayage de *k*.
4. **Si le résultat n'est pas satisfaisant** : veille et essai de nouvelles architectures apparues depuis MEDUSA.

---

## 1. Ce qui existe déjà

`src/tasks/htr/vlm_multiline_htr.py` est un squelette (décision 8 de l'annexe du doc 1), jamais
validé. Ce qu'il fait déjà et ce qu'il faut changer :

| Dans le squelette actuel | Ce qui doit changer |
|---|---|
| `_process_batch` découpe la page en blocs de `maxlines = 40` lignes **consécutives dans la liste globale**, sans respect des zones | Découper par blocs de **k** (paramètre, pas une constante en dur), **sans franchir une frontière de zone/`TextBlock`** (§3, Phase 0.2) |
| `_prepare_messages` empile **k images séparées** dans un seul message (`[{"type": "image", "image": img} for img in imgs]`) — c'est la stratégie « composite empilé » écartée par la décision 1 | Un seul crop : le rectangle englobant. `_prepare_messages` redevient une image unique + un prompt, comme `VLMLineHTR` |
| `_recognize_batch` sépare la sortie par `y.splitlines()` sans vérifier que le nombre de lignes obtenues correspond à *k* | Ajouter la vérification de compte + repli + dédoublonnage (§3, Phase 0.3) |
| `max_pixels` hérité tel quel de `BaseVLMHTR` (401 408, calibré pour une ligne) | Doit dépendre de *k* (§3, Phase 3.3) |
| Pas de support `{conventions}` ni `prompt_template` (présents dans `VLMLineHTR` mais pas portés ici) | À porter depuis `VLMLineHTR` (`prompt_convention.py`), pour rester cohérent avec le pipeline de production |

Le plan ci-dessous réécrit `VLMMultiLineHTRTask` par-dessus `BaseVLMHTR`, en réutilisant le plus
possible de `VLMLineHTR` (méthodes `_prepare_samples_with_conventions`, `train()`, structure de
config) plutôt que de dupliquer.

---

## 2. Vue d'ensemble des phases

| Phase | Contenu | Où ça s'exécute | Conditionne |
|---|---|---|---|
| 0 | Socle technique : crop, découpage en blocs, parsing de sortie, config | Local (CPU) | Tout le reste |
| 1 | Zero-shot MEDUSA sur crops multiligne | Serveur (inférence seule) | Go/no-go avant d'investir en finetuning |
| 2 | Finetune MEDUSA (LoRA poursuivi) sur multiligne, *k* fixe | Serveur / Jean Zay si gros volume | Donne une borne basse rapide avant le réentraînement complet |
| 3 | Réentraînement complet depuis Qwen3.5, balayage *k* = 1,2,4,8 | Jean Zay (volume ≈ MEDUSA) | Résultat principal : courbe CER(*k*) |
| 4 | Fenêtre roulante + vote | Local pour le code, serveur pour l'évaluation | Conditionnée : seulement si un *k* > 1 gagne en phase 3 |
| 5 | Veille nouvelles architectures | — | Conditionnée : seulement si le multiligne ne bat pas le ligne-à-ligne |

---

## 3. Détail par phase

> **Préambule à faire précéder par l'utilisateur devant chaque prompt ci-dessous** : tu travaille sur `docworkflow` (pipeline HTR : layout → segmentation de ligne → transcription), avec le contexte complet dans `01_synthese_segmentation_ligne.md`, `02_analyse_et_plan_action.md` et `03_plan_htr_multiligne.md` — à lire en cas de doute, sans déborder du scope de l'étape demandée.

### Phase 0 — Socle technique (local, CPU)

#### 0.1 Fonction de crop multi-ligne

Nouvelle méthode sur `VLMMultiLineHTRTask`, dans `src/tasks/htr/vlm_multiline_htr.py` (pas
dans la classe de base `BaseVLMHTR`, qui ne doit porter que ce qui est commun au
ligne-à-ligne et au multiligne) : un `_extract_multiline_image(image, boundaries)` qui prend
la liste des polygones des *k* lignes du bloc, calcule le rectangle englobant de leur union
(min/max sur tous les points, comme le fait déjà `_extract_line_image` pour une seule ligne),
crop **sans masque** (décision 2), et applique le même garde-fou de ratio d'aspect
(`max_ratio = 190`) que la fonction existante. Les deux primitives géométriques réutilisables
(rectangle englobant, padding de ratio) sont extraites dans `base_vlm_htr.py` sous forme de
fonctions partagées, pour que `_extract_line_image` et `_extract_multiline_image` les
appellent toutes les deux sans dupliquer de code ni mélanger la logique spécifique au
multiligne dans la classe de base.

> **Prompt Claude Code — 0.1**
> ```
> Dans src/tasks/htr/base_vlm_htr.py, extrais de _extract_line_image deux fonctions/méthodes
> partagées et réutilisables : une pour calculer le rectangle englobant d'un ensemble de points
> (actuellement calculé pour un seul polygone), une pour le padding de ratio d'aspect
> (max_ratio = 190). Ne touche pas au comportement de _extract_line_image par ailleurs.
>
> Dans src/tasks/htr/vlm_multiline_htr.py, ajoute une méthode _extract_multiline_image(self,
> image, boundaries) sur VLMMultiLineHTRTask (PAS dans base_vlm_htr.py — cette méthode est
> spécifique au multiligne et n'a rien à faire dans la classe de base). boundaries est une
> liste de polygones (chacun une liste de [x, y]). Calcule le rectangle englobant de l'UNION de
> tous les points de tous les polygones en réutilisant la fonction partagée extraite ci-dessus,
> crop l'image sur ce rectangle SANS appliquer de masque polygonal (contrairement à
> _extract_line_image qui masque en noir hors du polygone — ici on garde le rectangle brut tel
> quel), puis applique le même padding de ratio partagé.
>
>
> Ajoute un test unitaire dans le style des tests existants du projet.
> ```

#### 0.2 Découpage d'une page en blocs de *k* lignes

Règle : parcourir les lignes dans l'ordre de lecture déjà produit par `read_lines_geometry`
(`src/alto/alto_lines.py`), grouper par blocs de *k* **sans jamais faire chevaucher deux
`TextBlock`/zones** — un bloc s'arrête à la fin de sa zone même s'il contient moins de *k*
lignes. C'est le même principe que la restriction `MainZone` déjà en place pour la
segmentation (décision 2 de l'annexe du doc 1) : on ne mélange pas des zones dont les
conventions de mise en page diffèrent.

> **Prompt Claude Code — 0.2**
> ```
> Dans src/tasks/htr/vlm_multiline_htr.py, remplace le découpage actuel par maxlines=40 par un
> découpage paramétré par k = self.lines_per_crop (nouveau paramètre de config, défaut 4).
> Lis d'abord src/alto/alto_lines.py (read_lines_geometry) pour voir quelle information de zone
> est déjà disponible par ligne (TextBlock id, ou équivalent) et comment les lignes y sont
> ordonnées. Écris une fonction group_lines_into_blocks(lines, k) qui groupe les lignes
> consécutives par blocs de taille k, SANS faire chevaucher deux zones différentes : dès qu'on
> change de zone, on clôt le bloc courant (même s'il fait moins de k lignes) et on en commence
> un nouveau. Si l'information de zone n'existe pas encore dans les objets retournés par
> read_lines_geometry, propose le plus petit changement possible pour l'exposer (ne touche pas
> à la logique de parsing ALTO existante au-delà de ce qui est nécessaire). Ajoute un test
> unitaire avec un petit ALTO synthétique à deux zones pour vérifier qu'aucun bloc ne chevauche
> deux zones et que le dernier bloc d'une zone peut être plus petit que k.
> ```

#### 0.3 Parsing de la sortie : repli et dédoublonnage

Trois cas à couvrir, dans l'ordre où ils doivent être vérifiés après le `splitlines()` (ou,
si on bascule sur le jeton dédié, après le split sur ce jeton) :

1. **Nombre de lignes obtenu == k** : cas nominal, affectation directe aux IDs de ligne du
   bloc, dans l'ordre.
2. **Nombre de lignes obtenu != k** (repli) : logguer l'écart (compteur par run, utile pour
   juger la fiabilité d'un *k* donné), puis :
   - si moins de lignes que *k* : compléter par des chaînes vides pour les lignes manquantes
     (plutôt que de désaligner tout le reste du bloc) ;
   - si plus de lignes que *k* : tronquer aux *k* premières, **après** la passe de
     dédoublonnage (point 3), car un excédent est souvent un doublon dégénéré plutôt qu'un
     vrai contenu supplémentaire.
3. **Dédoublonnage** : si deux lignes consécutives de la sortie sont quasi identiques
   (distance d'édition normalisée en-dessous d'un seuil, ex. `< 0.1`), ne garder que la
   première occurrence avant de réaligner sur les *k* lignes attendues. **Garde-fou** : ne
   jamais appliquer cette règle au ground truth ni l'activer sur un corpus sans l'avoir
   d'abord mesurée (un document peut légitimement répéter une ligne — refrain, formule
   récurrente) ; logguer chaque collapse effectué pour audit a posteriori plutôt que de les
   appliquer silencieusement.

> **Prompt Claude Code — 0.3**
> ```
> Dans src/tasks/htr/vlm_multiline_htr.py, réécris _recognize_batch et la logique de parsing de
> sortie en t'appuyant sur group_lines_into_blocks (0.2) et sur une nouvelle fonction
> parse_multiline_output(raw_text, k) qui retourne (texts: list[str], n_mismatch: int,
> n_deduped: int). Règles à implémenter dans cet ordre :
> 1. Split raw_text sur '\n' (utilise jiwer ou une distance de Levenshtein simple — regarde si
>    le projet a déjà une dépendance d'édition de chaînes, sinon implémente une fonction courte,
>    pas de nouvelle dépendance pour ça) pour détecter des lignes consécutives quasi identiques
>    (distance d'édition normalisée par la longueur < 0.1) et ne garder que la première
>    occurrence de chaque paire détectée comme doublon, en comptant n_deduped.
> 2. Si len(texts) == k : ok, retourne tel quel.
> 3. Si len(texts) < k : complète avec des chaînes vides jusqu'à k, incrémente n_mismatch.
> 4. Si len(texts) > k : tronque aux k premières (le dédoublonnage de l'étape 1 a déjà dû
>    absorber l'essentiel du surplus), incrémente n_mismatch.
> Fais en sorte que n_mismatch et n_deduped remontent jusqu'aux résultats agrégés de
> _process_batch (par exemple dans le dict retourné par page), pour pouvoir plus tard calculer
> un taux de repli et de déduplication par valeur de k — c'est un indicateur de fiabilité qui
> sera réutilisé en phase 3. N'implémente PAS encore le jeton séparateur dédié (on reste sur
> '\n' pour l'instant, cf. §0 du plan) ; laisse un commentaire bref indiquant où il faudrait
> brancher un split alternatif si on en a besoin plus tard. Ajoute des tests unitaires pour les
> 4 cas (nominal, mismatch court, mismatch long, doublon consécutif détecté et non détecté sous
> le seuil).
> ```

#### 0.4 Réattribution aux IDs de ligne et écriture ALTO

Vérifier que `write_text_to_alto` (déjà utilisé par `VLMLineHTR` via un dict `line_id ->
result`) fonctionne sans changement une fois que `_process_batch` reconstruit ce même dict à
partir des blocs — c'est la même interface de sortie que le ligne-à-ligne (décision 8), donc
a priori aucun changement nécessaire côté ALTO.

> **Prompt Claude Code — 0.4**
> ```
> Dans src/tasks/htr/vlm_multiline_htr.py, termine _process_batch pour qu'il construise un dict
> line_id -> {'text': ..., 'confidence': ...} à partir des blocs de k lignes et de
> parse_multiline_output (0.3), exactement comme le fait déjà VLMLineHTRTask._process_batch
> dans src/tasks/htr/vlm_line_htr.py (regarde ce fichier comme référence directe). Réutilise
> copy_alto_without_text et write_text_to_alto sans les modifier. Porte aussi le support de
> {conventions} / prompt_template depuis vlm_line_htr.py (import de
> src/tasks/htr/prompt_convention.py) pour que VLMMultiLineHTR reste cohérent avec le pipeline
> de production. Vérifie avec un petit test d'intégration (ALTO + image synthétiques, modèle
> mocké pour ne pas charger de VLM) que le fichier ALTO de sortie contient bien un texte par
> TextLine dans le bon ordre, y compris quand un bloc a eu un repli (0.3).
> ```

#### 0.5 Plomberie config / CLI

Nouveaux paramètres de config pour `tasks.htr.config` (type `VLMMultiLineHTR`) :

- `lines_per_crop` (k, entier, défaut 4)
- `max_pixels` : garder le paramètre existant mais documenter qu'il doit être fourni
  explicitement pour k > 1 (pas de calcul automatique tant que la phase 3.3 n'a pas mesuré la
  bonne règle d'échelle)
- `rolling_window` : `false` par défaut (phase 4)
- `dedup_threshold` : seuil de distance d'édition pour 0.3, défaut 0.1

> **Prompt Claude Code — 0.5**
> ```
> Dans src/tasks/htr/vlm_multiline_htr.py (__init__ de VLMMultiLineHTRTask), ajoute la lecture
> des paramètres de config lines_per_crop (défaut 4), rolling_window (défaut False),
> dedup_threshold (défaut 0.1), en suivant le style déjà utilisé pour les autres paramètres
> (config.get(...)). Crée un fichier configs/medusa/MedusaMultiLine-4B.yml sur le modèle de
> configs/medusa/Medusa0.2Line-4B.yml (lis-le d'abord), avec tasks.htr.type: VLMMultiLineHTR,
> lines_per_crop: 4, et un data.test à TO FILL IN. N'ajoute aucune valeur de max_pixels
> "automatique" pour l'instant — laisse le max_pixels hérité de BaseVLMHTR par défaut et mets
> un commentaire indiquant que la bonne règle d'échelle sera déterminée en phase 3.3 du plan
> (03_plan_htr_multiligne.md).
> ```

**Critère de sortie de la phase 0** : `docworkflow -c configs/medusa/MedusaMultiLine-4B.yml
predict -t htr -d test` tourne de bout en bout sur un petit jeu de pages (local, modèle MEDUSA
existant chargé tel quel, k=4) et produit un ALTO exploitable. C'est aussi le point d'entrée de
la phase 1.

---

### Phase 1 — Zero-shot : MEDUSA tel quel sur des crops multiligne

Objectif : savoir, **avant d'investir dans un entraînement**, si MEDUSA dégrade
gracieusement ou s'effondre quand on lui donne un crop à *k* lignes au lieu d'une. C'est un
test bon marché (inférence seule) qui peut à lui seul motiver ou désamorcer les phases 2-3.

- Corpus : dev set ICDAR-CMMHWR (§2.2 du doc 2), polygones GT, pas le test set gelé MEDUSA
  (réservé au chiffre final).
- *k* ∈ {1, 2, 4, 8}, *k*=1 étant le CER ligne-à-ligne déjà publié, qui sert de référence.
- `max_pixels` : garder le réglage MEDUSA d'origine inchangé (401 408) à ce stade — le but
  est justement de voir l'effet brut de la compression de résolution par ligne, pas de le
  compenser.
- Métrique : CER par ligne après réattribution (le mécanisme de score existant,
  `task.score` dans `src/tasks/htr/base_htr.py`, compare déjà texte prédit/texte GT par
  TextLine — à vérifier qu'il fonctionne sans changement une fois l'ALTO multiligne produit).
- Plus, pour chaque *k* : taux de repli (`n_mismatch`) et taux de déduplication
  (`n_deduped`) de la phase 0.3, qui disent si le modèle « tient » le format à ce *k* avant
  même de regarder le texte.

> **Prompt Claude Code — Phase 1 (à lancer sur le serveur, après push)**
> ```
> Contexte : src/tasks/htr/vlm_multiline_htr.py vient d'être réécrit (voir
> 03_plan_htr_multiligne.md, phase 0) pour produire un crop rectangle englobant de k lignes
> consécutives et parser la sortie avec repli/dédoublonnage. Le modèle MEDUSA0.2Line-4B
> (configs/medusa/Medusa0.2Line-4B.yml) n'a PAS été réentraîné pour le multiligne : il s'agit
> ici de tester tel quel ce modèle ligne-à-ligne sur des crops multiligne, en zero-shot.
>
> Tâche :
> 1. Crée configs/medusa/MedusaMultiLine-4B_zeroshot_k{2,4,8}.yml (3 fichiers), dérivés de
>    configs/medusa/MedusaMultiLine-4B.yml, avec model_name et max_pixels identiques à
>    Medusa0.2Line-4B.yml (NE PAS changer max_pixels à ce stade), lines_per_crop respectivement
>    2, 4, 8, et data.test pointant vers le dev set ICDAR-CMMHWR (demande le chemin exact si tu
>    ne le trouves pas dans les configs existantes de prédiction du dev set).
> 2. Lance `docworkflow -c <config> predict -t htr -d test` pour chacun des 3 k, puis
>    `docworkflow -c <config> score -t htr --pred <sortie> --gt <dev_set_gt>` (vérifie la
>    syntaxe exacte de la commande score dans src/cli/score.py et comment elle est invoquée
>    ailleurs dans le projet, par exemple dans scripts/).
> 3. Relève pour chaque k : le CER agrégé, n_mismatch et n_deduped totaux (ajoute-les au besoin
>    à la sortie de la commande score si ce n'est pas déjà exposé — sinon logue-les séparément
>    depuis predict).
> 4. Compare au CER ligne-à-ligne déjà publié pour Medusa0.2Line-4B (k=1, résultats existants
>    dans results/ ou jz_results/ — cherche-les).
> 5. Produis un tableau k / CER / n_mismatch / n_deduped, et une inspection visuelle de 10 à 20
>    exemples (sauvegarde les crops + prédictions + GT pour inspection manuelle, par exemple
>    dans outputs/phase1_zeroshot_multiligne/).
> Ne lance aucun entraînement dans cette tâche — c'est de l'inférence seule sur un modèle déjà
> entraîné.
> ```

**Critère de décision** (go/no-go avant la phase 2) : si le CER se dégrade de façon
catastrophique dès *k*=2 (ex. > 2x le CER ligne-à-ligne) avec un taux de repli élevé, c'est
cohérent avec la raison d'échec du page-level déjà documentée (§6 de l'article MEDUSA, citée
au doc 2 — « À éviter ») et renforce l'intérêt d'un réentraînement dédié plutôt que d'un usage
zero-shot en production. Si la dégradation est modérée et régulière en *k*, cela donne une
première estimation (optimiste ou pessimiste à affiner) de la courbe visée en phase 3.

---

### Phase 2 — Finetune MEDUSA (LoRA poursuivi) sur multiligne

Objectif : une borne basse rapide, moins coûteuse que le réentraînement complet de la phase 3,
pour voir si partir de MEDUSA (qui sait déjà transcrire une ligne) accélère l'apprentissage du
format multiligne.

- *k* fixe pour cette phase (proposition : *k*=4, valeur médiane du balayage prévu en phase 3 ;
  pas besoin de balayer ici, c'est un test de faisabilité, pas la mesure de référence).
- Données d'entraînement : les mêmes lignes GT que MEDUSA (643 k lignes), regroupées en blocs
  de *k* selon la règle de la phase 0.2 (pas de passage inter-zone).
- Mécanisme de poursuite du LoRA : déjà supporté par l'infrastructure existante — dans
  `VLMLineHTR.train()`, passer `base_model` déclenche un chargement de l'adaptateur MEDUSA
  existant, sa fusion dans le modèle de base, puis l'entraînement d'un nouveau LoRA sur ce
  modèle fusionné (cf. `configs/hidden/jeanzay/jeanzay_T3-2.yml`, qui fait exactement ça pour
  continuer un entraînement ligne-à-ligne). Même mécanisme à réutiliser ici, pas de nouveau
  code d'entraînement : `model_name` = adaptateur MEDUSA0.2Line-4B, `base_model` = Qwen3.5-4B.

> **Prompt Claude Code — Phase 2.1 (préparation des données, local)**
> ```
> Dans src/tasks/htr/vlm_multiline_htr.py, écris _prepare_training_data_lines et
> _format_conversation pour VLMMultiLineHTRTask en t'inspirant de VLMLineHTRTask
> (src/tasks/htr/vlm_line_htr.py) plutôt que de la version actuelle du squelette (qui empile
> des images séparées — décision écartée, voir 03_plan_htr_multiligne.md §0). Chaque exemple
> d'entraînement doit correspondre à UN bloc de k lignes (utilise group_lines_into_blocks de
> la phase 0.2), avec un unique crop rectangle englobant (_extract_multiline_image de la phase
> 0.1) en entrée, et en sortie assistant le texte des k lignes séparé par '\n' dans l'ordre de
> lecture du bloc. Porte aussi le support prompt_template / {conventions} comme
> VLMLineHTRTask._prepare_samples_with_conventions. Ajoute un test qui vérifie, sur un petit
> ALTO synthétique à 6 lignes et k=4, qu'on obtient bien 2 exemples d'entraînement (un bloc de
> 4, un bloc de 2) avec le bon texte concaténé. Ne lance aucun entraînement, uniquement la
> préparation des données.
> ```

> **Prompt Claude Code — Phase 2.2 (config + lancement, serveur/Jean Zay)**
> ```
> Contexte : src/tasks/htr/vlm_multiline_htr.py a maintenant une méthode train() à écrire pour
> VLMMultiLineHTRTask — reprends celle de VLMLineHTRTask (src/tasks/htr/vlm_line_htr.py) presque
> telle quelle (c'est la même mécanique Unsloth/LoRA/SFTTrainer), en l'adaptant seulement pour
> utiliser les _prepare_training_data_lines / _format_conversation multiligne de la phase 2.1
> de 03_plan_htr_multiligne.md.
>
> Crée configs/hidden/jeanzay/jeanzay_multiligne_k4_continue.yml sur le modèle de
> configs/hidden/jeanzay/jeanzay_T3-2.yml (lis-le d'abord) : même mécanisme de poursuite de
> LoRA (model_name = chemin de l'adaptateur MEDUSA0.2Line-4B déjà entraîné, base_model =
> chemin Qwen3.5-4B sur Jean Zay), tasks.htr.type: VLMMultiLineHTR, lines_per_crop: 4, mêmes
> données d'entraînement que MEDUSA (vérifie le chemin exact dans jeanzay_T3-2.yml ou les
> configs voisines). Documente dans le message de retour la commande sbatch / le script de
> lancement à utiliser (cherche un script existant du type scripts/train_*.sh ou les
> instructions Jean Zay déjà utilisées pour MEDUSA, ne l'invente pas).
>
> N'exécute PAS l'entraînement toi-même (pas de GPU local) : prépare la config et le code
> jusqu'au point où l'utilisateur n'a plus qu'à pousser et lancer sur Jean Zay.
> ```

**Critère de sortie** : CER (k=4, finetuné) comparé au CER zero-shot (k=4, phase 1) et au CER
ligne-à-ligne (k=1). Sert surtout à calibrer les attentes avant la phase 3, qui est la mesure
de référence.

---

### Phase 3 — Réentraînement complet depuis Qwen3.5, balayage de *k*

C'est le résultat principal (cf. doc 2 §6, résultat visé n°2 : « la courbe contexte /
hallucination en fonction du nombre de lignes »).

#### 3.1 Jeux d'entraînement

Un jeu par *k* ∈ {2, 4, 8}, construits comme en phase 2.1. *k*=1 est déjà MEDUSA
ligne-à-ligne publié, utilisé comme contrôle sans nouveau run.

#### 3.2 Entraînements

Un LoRA **depuis Qwen3.5 base**, pas depuis l'adaptateur MEDUSA (c'est la différence avec la
phase 2 : repartir du même pré-entraîné, pas poursuivre un LoRA ligne-à-ligne qui a déjà
spécialisé le modèle sur le format une-ligne). Reprendre le gabarit des configs Jean Zay
d'entraînement initial de MEDUSA (`jeanzay_T1*.yml`, sans `base_model` pointant vers un
adaptateur existant) plutôt que celui de la phase 2 (`jeanzay_T3-2.yml`).

#### 3.3 Budget de pixels — l'arbitrage central, à mesurer

Le doc 2 est explicite : « plus de contexte contre moins de pixels par ligne... à mesurer, pas
à supposer ». Pour ne pas confondre l'effet de *k* avec l'effet de la résolution, prévoir **au
moins deux réglages de `max_pixels` par valeur de *k*** si le budget GPU le permet :

- **proportionnel** : `max_pixels(k) = k × 401408`, qui maintient la résolution par ligne
  constante (le crop multiligne a simplement plus de jetons image au total) ;
- **fixe** : `max_pixels` identique à MEDUSA (401 408) quel que soit *k*, qui mesure l'effet
  pur de la compression de résolution.

Si le budget GPU est contraint, prioriser le réglage **proportionnel** seul (c'est la lecture
la plus proche de « plus de contexte sans perdre de résolution », donc la plus susceptible de
montrer un bénéfice du contexte s'il existe), et noter le réglage fixe comme extension.

#### 3.4 Évaluation

- Développement/sélection sur le **dev set ICDAR-CMMHWR** (comme en phase 1).
- Chiffre publiable uniquement sur le **test set gelé MEDUSA** (63 pages / 36 documents),
  et seulement pour le ou les *k* retenus à l'issue du dev — pas pour balayer.
- Reporter, comme le fait déjà le doc 2 pour le balayage YOLO (§1.5) : CER par *k*, taux de
  repli et de déduplication par *k*, et si possible une décomposition par langue (Figure 4 de
  l'article MEDUSA, citée au doc 2 §4.1 : effet attendu maximal sur les langues peu
  représentées).
- Vigilance sur le bruit de mesure (leçon du doc 2 §1.5, « étages 2 et 4 ») : si deux valeurs
  de *k* sont à moins de quelques points de CER l'une de l'autre, ne pas trancher sans au
  moins une graine supplémentaire sur le *k* candidat.

> **Prompt Claude Code — Phase 3 (préparation, local ; lancement, Jean Zay)**
> ```
> Contexte : 03_plan_htr_multiligne.md décrit un balayage k = {2, 4, 8} pour VLMMultiLineHTR,
> en réentraînement complet depuis Qwen3.5 base (PAS une poursuite de l'adaptateur MEDUSA —
> voir la différence avec jeanzay_T3-2.yml en phase 2). Les jeux de données par k réutilisent
> _prepare_training_data_lines de VLMMultiLineHTRTask (phase 2.1) et group_lines_into_blocks
> (phase 0.2).
>
> Tâche :
> 1. Crée trois configs Jean Zay, une par k ∈ {2,4,8}, sur le modèle du gabarit d'entraînement
>    initial de MEDUSA (cherche dans configs/hidden/jeanzay/ celui qui entraîne depuis Qwen3.5
>    base sans adaptateur existant — probablement jeanzay_T1*.yml, vérifie), avec
>    tasks.htr.type: VLMMultiLineHTR, lines_per_crop: k, et max_pixels = k * 401408 (réglage
>    proportionnel, voir §3.3 du plan). Même split train/valid que MEDUSA.
> 2. Documente, pour chaque config, la commande de lancement Jean Zay (sbatch ou équivalent),
>    en reprenant celle déjà utilisée pour MEDUSA sans l'inventer.
> 3. Prépare aussi, dans le même mouvement, les configs de prédiction + score sur le dev set
>    ICDAR-CMMHWR pour chaque k entraîné, et une config finale sur le test set gelé MEDUSA pour
>    le k qui sera retenu une fois le dev set évalué (laisse le choix du k en paramètre, ne le
>    fixe pas en dur).
> N'exécute aucun entraînement : prépare tout jusqu'au point où l'utilisateur lance sur Jean Zay.
> ```

**Critère de sortie** : courbe CER(*k*) sur le dev set, décision du *k* retenu (ou constat que
*k*=1 reste le meilleur, auquel cas la phase 4 ne se déclenche pas — cf. tableau §2), puis
confirmation sur le test set gelé.

---

### Phase 4 — Fenêtre roulante + vote (conditionnelle)

**Déclencheur** : seulement si un *k* > 1 bat le CER ligne-à-ligne en phase 3. Si *k*=1 reste
optimal, la fenêtre roulante n'a pas d'objet (elle sert à lisser le compromis contexte/bruit
d'un *k* qui, par hypothèse, apporte déjà un gain).

- Fenêtre de taille *k* (le *k* retenu en phase 3), pas *s* < *k* (proposition : *s* = 1, chaque
  ligne est vue dans *k* fenêtres différentes — à ajuster si trop coûteux ; *s* = k/2 en repli
  si *s*=1 est hors budget).
- Agrégation par vote au niveau caractère après alignement (ROVER), comme proposé au doc 2 —
  pas par log-probabilité dans un premier temps (la génération actuelle est en `do_sample=False`
  sans scores retournés ; l'ajouter serait un changement plus intrusif du code de génération,
  à ne faire que si le vote caractère déçoit).
- Le désaccord entre fenêtres pour une même ligne (ex. proportion de caractères où les
  hypothèses divergent) est stocké comme signal de confiance, dans l'attribut `CONFIDENCE`
  déjà utilisé côté segmentation (`src/alto/alto_lines.py`, `CONFIDENCE_ATTR`) — réutiliser le
  même attribut côté HTR plutôt qu'en inventer un nouveau.
- Option CLI explicite (`rolling_window: true/false`, déjà prévue en 0.5) pour comparer
  systématiquement avec et sans, en coût (passes supplémentaires) et en CER.

> **Prompt Claude Code — Phase 4**
> ```
> Contexte : la phase 3 de 03_plan_htr_multiligne.md a confirmé qu'un k > 1 (valeur à préciser
> ici) bat le ligne-à-ligne. Implémente la fenêtre roulante pour VLMMultiLineHTRTask :
> 1. Dans _process_batch, quand self.rolling_window est True, génère pour chaque ligne
>    jusqu'à k hypothèses (une par fenêtre de taille k et de pas s=1 qui la contient), au lieu
>    d'un découpage en blocs non chevauchants (group_lines_into_blocks de la phase 0.2 reste
>    utilisé pour le mode non-roulant, inchangé).
> 2. Implémente une fonction rover_vote(hypotheses: list[str]) -> tuple[str, float] qui aligne
>    les hypothèses (cherche si une dépendance d'alignement de séquences est déjà disponible
>    dans le projet — jiwer est déjà une dépendance, regarde ce qu'elle expose pour l'alignement
>    avant d'en ajouter une nouvelle) et vote caractère par caractère (majorité), en retournant
>    aussi un score de désaccord (proportion de caractères où le vote n'est pas unanime).
> 3. Écris ce score de désaccord dans l'attribut CONFIDENCE de l'ALTO de sortie, en réutilisant
>    CONFIDENCE_ATTR de src/alto/alto_lines.py (ne crée pas un nouvel attribut).
> 4. Ajoute un test unitaire sur rover_vote (3-4 hypothèses synthétiques avec un désaccord
>    localisé, vérifier le texte voté et le score de désaccord).
> Ne lance aucune évaluation GPU : c'est un travail de code et de tests locaux. L'évaluation
> avec/sans fenêtre roulante sur le dev set est une étape séparée, à lancer sur le serveur une
> fois ce code validé.
> ```

Ensuite, séparément (sur le serveur) : comparer CER et coût (nombre de passes, temps) avec et
sans fenêtre roulante, et vérifier la corrélation entre le score de désaccord et le CER réel
par ligne (calibration) — c'est l'information utile à l'objectif 3 (frontend) mentionnée au
doc 2 §4.1.

---

### Phase 5 — Si insatisfaisant : nouvelles architectures (ouverte)

**Déclencheur** : le CER multiligne (meilleur *k*, avec ou sans fenêtre roulante) ne descend
pas sous le CER ligne-à-ligne, malgré le balayage de *k* et des deux réglages de pixels. Pas de
plan détaillé à ce stade — seulement un rappel du critère de déclenchement et le constat,
déjà fait dans le message utilisateur, qu'il faudra alors regarder ce qui a émergé depuis
Qwen3.5/MEDUSA plutôt que de ré-essayer les mêmes leviers.

---

## 4. À éviter

- **Comparer un CER zero-shot (phase 1) à un CER post-finetuning (phases 2-3) sur des corpus
  différents.** Garder le dev set ICDAR-CMMHWR comme terrain commun à toutes les phases, le
  test set gelé MEDUSA réservé au chiffre final.
- **Confondre l'effet de *k* et l'effet du budget de pixels.** Sans le réglage proportionnel
  de `max_pixels` (§3.3), une baisse de CER à *k* élevé peut n'être qu'un artefact de
  résolution, et une hausse peut cacher un vrai effet de contexte noyé par la perte de
  résolution.
- **Trancher le balayage de *k* sur un écart de quelques points de CER sans graine
  supplémentaire**, par analogie directe avec la leçon du doc 2 (§1.5, étages 2 et 4, tranchés
  à tort sur du bruit).
- **Dédoublonner aveuglément.** Le garde-fou de la phase 0.3 (logguer, ne jamais appliquer au
  GT, ne pas l'activer sans l'avoir mesuré) doit rester actif même si le taux de déduplication
  semble faible sur le dev set — un corpus différent peut légitimement répéter des lignes.
- **Lancer un entraînement ou une inférence en volume sur cette machine.** Toute étape au-delà
  de la préparation de code/config et de tests unitaires sur fixtures synthétiques doit être
  poussée et exécutée sur le serveur ou Jean Zay.
- **Activer la fenêtre roulante avant que la phase 3 ait confirmé qu'un *k* > 1 apporte un
  gain.** Elle n'a de sens que pour lisser un *k* déjà gagnant, pas pour sauver un *k* perdant.

---

## 5. Récapitulatif

| Phase | Déclencheur | Coût approximatif | Sortie |
|---|---|---|---|
| 0. Socle | Immédiat | Local, quelques heures de dev | `VLMMultiLineHTR` fonctionnel de bout en bout, k configurable |
| 1. Zero-shot | Après phase 0 | Inférence seule, serveur, quelques heures | Go/no-go, première courbe CER(k) non entraînée |
| 2. Finetune MEDUSA | Après phase 1 | 1 run LoRA, serveur/Jean Zay | Borne basse rapide, k=4 |
| 3. Réentraînement complet | Après phase 0 (indépendante de 1-2 pour démarrer les jeux de données) | 3 runs LoRA depuis Qwen3.5, Jean Zay (ordre de grandeur MEDUSA) | Courbe CER(k) de référence, k retenu |
| 4. Fenêtre roulante | Si un k>1 gagne en phase 3 | Code local + évaluation serveur | CER avec/sans fenêtre, signal de confiance calibré |
| 5. Nouvelles architectures | Si rien ne bat le ligne-à-ligne | Ouvert | — |
