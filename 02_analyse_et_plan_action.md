# Analyse et plan d'action — Segmentation de ligne

> Document opérationnel, **état au 12 octobre 2026**. Il décrit la situation présente et ce qu'il
> faut faire ensuite ; il ne conserve pas la trace des itérations qui y ont mené.
>
> Sources : le benchmark de cinq modèles figés (`scripts/benchmark_line_models.sh`), le **balayage
> contrôlé d'entraînement** du §1.5 (`scripts/train_line_sweep.sh`, 12 entraînements sur RTX 3090,
> `results/line_sweep_20260930_114414/`), le code de `DocWorkflow`, et les décisions consignées en
> annexe du document 1. Les chiffres du §1.4 sont reproductibles par `outputs/phaseA_faux_positifs/`.

## En un écran

**Le segmenteur est choisi : `yolo26s` en détection, 1280 px à l'entraînement et à l'inférence,
100 époques, `mosaic` ≤ 0,2.** Issu d'un balayage en ablation par étages (§1.5), il prend la tête du
benchmark sur six colonnes sur sept — `map50-95` 0,706 et F1 0,867, contre 0,682 et 0,829 pour le
meilleur Kraken — et le §1.4 montre qu'il est à ~1,5 point du plafond structurel d'un modèle à boîtes
sur ZoneMapAlt. La marge était bien dans l'entraînement : 0,060 de `map50-95` séparent le même
backbone mal et bien réglé, soit près de deux fois l'écart qui séparait alors YOLO de Kraken sur
cette métrique.

**Trois choses ne sont pas réglées, et elles commandent la suite :**

| Verrou | Pourquoi il bloque | Section |
|---|---|---|
| **Pas de métrique baseline (cBAD)** | La cible retenue n'est mesurée par rien, et le mAP de boîtes mesure *exactement* l'objectif du détecteur — il ne peut donc pas arbitrer det vs seg vs OBB | §2.1 |
| **Plancher de bruit non mesuré** | Deux étages du balayage se sont joués à ≤ 0,004 de `map50-95` ; 3 graines (≈ 4 h) les établiraient | §1.5 |
| **OBB entièrement à faire** | Aucun dataset, `result.obb` jamais lu, et `polygon_to_baseline` divise par deux la pente d'un quadrilatère | §3.1 |

---

## 1. Où en est la segmentation de ligne

### 1.1 Protocole de mesure

| | |
|---|---|
| Corpus d'évaluation | `../data/ICDAR_CMMHWR_original_data` — **286 pages, 53 documents, 11 562 lignes**, gelé |
| Candidats | 2 Kraken (`.mlmodel`), 3 YOLO figés, **+ 12 YOLO réentraînés** (§1.5) |
| Résolution | résolution d'entraînement de chaque modèle, jamais une autre (§1.5) |
| Protocole de zone | `restrict_to_layout: true` pour tous |
| Métriques | mAP@50/75/50-95, P/R/F1 au point d'opération (IoU 0,75), ZoneMapAlt |
| Trace | wandb, projets `LS-comparison` et `LS-training` |

Le protocole de zone mérite d'être explicité, parce que c'est la seule façon de comparer les deux familles honnêtement. Kraken reçoit les zones de layout **en entrée d'inférence** et ne peut structurellement pas prédire hors zone ; YOLO voit la page entière et n'est confronté au layout qu'à l'écriture de l'ALTO. 
Avec `restrict_to_layout: false`, les lignes YOLO tombant hors zone survivent en pseudo-zones sans `TAGREFS`, que le filtre `IGNORED_ZONE_TYPES` du scoring ne retire
pas alors qu'il retire les lignes de référence correspondantes : chaque note marginale correctement détectée est alors comptée en faux positif. `true` rétablit la symétrie **et** correspond à l'usage
réel en production.

Vérifié, puisque le scoring en dépend : les 11 562 lignes du corpus portent toutes un attribut
`BASELINE`, et 99,97 % portent en plus un polygone.

### 1.2 Les chiffres

Précision, rappel et F1 sont pris au point d'opération, à IoU 0,75 — donc lisibles comme des taux,
1,00 étant parfait. `zonemap/score` est une **pénalité** : plus bas est meilleur.

| Modèle | map50 | map75 | map50-95 | P | R | F1 | zonemap |
|---|---|---|---|---|---|---|---|
| **yolo26s det, 1280 px, 100e** (§1.5) | **0,888** | **0,860** | **0,706** | **0,866** | **0,869** | **0,867** | 38,8 |
| yolo26s seg, 1280 px, 100e (§1.5) | 0,878 | 0,724 | 0,593 | 0,771 | 0,783 | 0,775 | 28,0 |
| kraken `baseline_ms_medieval` | 0,849 | 0,779 | 0,682 | 0,827 | 0,833 | 0,829 | 15,4 |
| kraken `blla` | 0,818 | 0,732 | 0,674 | 0,778 | 0,823 | 0,797 | **13,1** |
| yolo26s det 640 px 50e (figé) | 0,863 | 0,806 | 0,646 | 0,791 | 0,835 | 0,811 | 45,0 |
| `LINES-SEG` (externe) | 0,879 | 0,428 | 0,465 | 0,559 | 0,551 | 0,555 | 28,1 |
| yolo26n seg 10e (figé) | 0,319 | 0,060 | 0,119 | 0,233 | 0,157 | 0,173 | 85,4 |

Décomposition ZoneMapAlt, en aire (fraction de l'aire totale de référence) et en comptages moyens
par document :

| Modèle | match | miss | faux pos. | n_miss | n_faux pos. | n_merge | n_split | n_multiple |
|---|---|---|---|---|---|---|---|---|
| **yolo26s det, 1280 px, 100e** | 0,969 | 0,025 | 0,358 | 2,9 | 3,0 | 1,9 | 4,3 | 45,1 |
| yolo26s seg, 1280 px, 100e | 0,822 | 0,140 | 0,106 | 5,2 | 4,0 | 0,8 | 5,6 | 84,0 |
| kraken `baseline_ms_medieval` | 0,893 | 0,096 | 0,049 | 4,2 | 10,1 | 1,3 | 7,0 | 20,2 |
| kraken `blla` | 0,904 | 0,084 | **0,035** | 2,2 | 9,8 | 2,0 | 6,3 | 15,8 |
| yolo26s det 640 px 50e (figé) | 0,939 | 0,051 | 0,389 | 9,4 | 9,2 | 3,5 | 16,8 | 53,3 |
| `LINES-SEG` (externe) | 0,809 | 0,150 | 0,086 | 2,7 | 3,0 | 0,8 | 1,2 | 66,4 |
| yolo26n seg 10e (figé) | 0,386 | 0,523 | 0,241 | 69,8 | 20,1 | 28,3 | 54,3 | 42,2 |
| *oracle de boîtes (référence)* | *1,000* | *0,000* | *0,333* | *0,0* | *0,0* | *0,0* | *0,0* | *0,0* |

La dernière ligne est un détecteur fictif, non un modèle : une prédiction par ligne de référence, égale à la **boîte englobante de son polygone**. Rappel et précision sont parfaits par construction, et pourtant il paie 0,333 d'aire de faux positifs et un `zonemap/score` de **33,3**. C'est le coût
du format de sortie, et c'est la clé de lecture du §1.4.

### 1.3 Le classement dépend de la métrique, et il faut savoir pourquoi

Deux colonnes du §1.2 ne sont pas des comparaisons loyales entre les deux familles, et il faut le
savoir avant de les citer.

**Les colonnes `map*` avantagent YOLO, parce que lui seul produit des confiances.** Kraken n'en
produit pas — `read_lines` leur attribue 1,0 — et l'AP intègre une courbe précision–rappel obtenue
en faisant varier le seuil, donc des confiances calibrées améliorent l'AP **à géométrie constante**.
Un run de contrôle, confiances de yolo26s écrasées à 1,0, chiffre l'effet : `map50` −0,061,
`map75` −0,065, `map50-95` −0,052, et **P/R/F1 comme ZoneMapAlt inchangés** — ce qui est attendu,
ni le point d'opération ni ZoneMapAlt ne dépendent d'un classement par confiance. À confiances
égalisées, le yolo26s figé passe dernier sur `map50`. L'avantage est réel en production — une
confiance exploitable est un livrable en soi — mais il ne dit rien de la qualité géométrique.

**La colonne ZoneMapAlt pénalise YOLO, parce que lui seul produit des rectangles.** C'est le §1.4.

**Kraken garde l'avantage là où l'on mesure une propriété dont on n'a pas l'usage.** Son avantage
résiduel se concentre sur les seuils d'IoU extrêmes (0,85 / 0,90 / 0,95) où une boîte rectangulaire
ne peut plus épouser une ligne inclinée. Or la cible retenue est la baseline (décision 3) et le crop
envoyé à MEDUSA est de toute façon un rectangle redressé.

### 1.4 ZoneMapAlt mesure un format de sortie

**L'essentiel de la pénalité ZoneMapAlt d'un détecteur de boîtes vient du fait qu'il rend des
rectangles.** L'oracle du §1.2 le fixe : un détecteur parfait prédisant exactement la boîte
englobante de chaque polygone de référence — rappel et précision parfaits, zéro manque, zéro faux
positif compté — obtiendrait tout de même `zonemap/score` = **33,3** et `false_alarm` = **0,333**.
En retirant ce plancher au premier ordre :

| Modèle | zonemap | plancher de format | imputable au modèle |
|---|---|---|---|
| yolo26s det 1280 px 100e | 38,8 | 33,3 | **≈ 5,5** |
| yolo26s det 640 px figé | 45,0 | 33,3 | ≈ 11,6 |
| kraken `blla` | 13,1 | 0 (polygones) | 13,1 |

La soustraction est une approximation — le débordement réel n'est pas exactement celui de l'oracle —
mais l'ordre de grandeur ne laisse pas de place au doute : **une fois le format neutralisé, le modèle
du §1.5 est le plus propre du benchmark**, à ~1,5 point de son plafond structurel. Aucun réglage
d'entraînement en formulation boîte ne récupérera les 33,3 restants.

**Conséquence : `zonemap/score` ne peut pas départager un détecteur de boîtes d'un segmenteur de
polygones.** Il reste valide au sein d'une même famille, et pour lire fusions, scissions et manques,
qui ne dépendent pas du format.

**Piège de lecture.** `_calc_detection_raw()` (`src/utils/zonemap.py`) somme l'aire de **tous** les
groupes de faux alarme, mais n'en *compte* que les prédictions pleinement non appariées : le
débordement d'une prédiction par ailleurs correctement appariée alimente l'aire sans alimenter le
comptage. Sur le yolo26s figé, 11 397 résidus de prédictions appariées portent 88,7 % de l'aire pour
0,46 ligne en moyenne, contre 452 faux positifs réels portant 11,3 % de l'aire pour 1,48 ligne.
**Diviser `false_alarm` par `n_false_alarm` n'a donc pas de sens** ; même dissymétrie pour `miss` /
`n_miss`.

#### Ce que sont les faux positifs de YOLO

Décompte exhaustif des 452 faux positifs réels du yolo26s figé, par ce qu'ils recouvrent :

| Famille | nombre | part de l'aire |
|---|---|---|
| **doublon sur une ligne déjà détectée** | 301 (66,6 %) | 72,8 % |
| **doublon à cheval sur deux lignes ou plus** | 83 (18,4 %) | 25,8 % |
| zone vide ou décor (lettrine, filet, ornement) | 68 (15,0 %) | 1,5 % |

**85 % sont des doublons** — un problème de NMS, pas de résolution ; les 15 % restants, décor et
marges, ne pèsent rien. Le plus grand faux positif du corpus couvre **6,0 lignes**, trois dépassent
3 lignes, aucun n'atteint 10 lignes (médiane 1,46, p90 1,81) : il n'y a pas de « boîtes-paragraphes »
sur ce corpus. Vérifié visuellement sur les vingt plus grands
(`outputs/phaseA_faux_positifs/crops_top20/`), normalisés par l'aire moyenne d'un groupe apparié de
la même page — trier par aire absolue concentre le top 20 dans le seul document à images de 104 Mpx.

Le modèle du §1.5 a largement absorbé le problème sans y toucher : `n_faux pos.` **3,0 contre 9,2**
et `n_split` **4,3 contre 16,8**.

#### Le seuil de confiance n'est pas le levier

Balayage sur `CONFIDENCE` du yolo26s figé, sans réinférence (filtrage des `TextLine` puis rescoring —
44,9 au rescoring contre 45,0 au benchmark, même modèle, écart d'arrondi) :

| seuil | P | R | F1 | zonemap | FA aire | n_FA | miss aire | n_miss |
|---|---|---|---|---|---|---|---|---|
| 0,25 (sortie brute) | 0,791 | **0,834** | 0,811 | 44,9 | 0,387 | 9,22 | 0,053 | 9,5 |
| **0,40** | 0,838 | 0,800 | **0,817** | **41,6** | 0,317 | 3,59 | 0,086 | 18,9 |
| 0,60 | 0,871 | 0,716 | 0,780 | 43,9 | 0,247 | 0,69 | 0,170 | 41,0 |
| 0,80 | **0,890** | 0,553 | 0,662 | 55,4 | 0,172 | 0,06 | 0,359 | 94,3 |
| 0,90 | 0,842 | 0,197 | 0,301 | 82,1 | 0,059 | 0,02 | 0,752 | 148,0 |

Le seuil fait exactement ce qu'on attend de lui **sur les doublons** — `n_FA` tombe de 9,2 à 0,7 dès
0,60 — et pourtant `zonemap` ne descend qu'à 41,6 avant de remonter : l'aire de faux positifs qui
subsiste après élimination des doublons est le débordement des rectangles appariés, que le seuil
n'atteint pas par construction, et l'aire de manques la remplace plus vite qu'elle ne décroît.

**0,40 est le meilleur compromis sur ce modèle-là**, et c'est un réglage de production défendable,
pas une correction du diagnostic. Mais il a été calibré sur un modèle dont `n_FA` valait 9,2 ; celui
du §1.5 est à 3,0 sans aucun filtrage. **Le seuil est donc à recalibrer avant d'être reconduit**, et
il ne doit jamais servir à départager deux modèles.

*Réserve sur les deux dernières lignes :* `_score_batch` fait `continue` sur toute page sans
prédiction **avant** le calcul ZoneMapAlt, donc les pages entièrement vidées par le filtre sortent du
score au lieu d'être comptées en manques. Aux seuils 0,80 et 0,90, le tableau est optimiste.

### 1.5 Le balayage contrôlé d'entraînement

**Acquis préalable, qui fixe le protocole :** monter `img_size` à la seule prédiction, sur modèle
figé, dégrade fortement — jusqu'à −41 % de `map50-95` pour yolo26s. Entraînement et inférence
doivent partager la même résolution, et c'est pourquoi chaque modèle est évalué à la sienne (§1.1).

Le balayage teste donc la résolution **à l'entraînement**, en ablation par étages : une variable à
la fois, chaque étage fixant le gagnant du précédent, tous évalués sur le corpus gelé.
**12 entraînements, 10,7 h GPU sur RTX 3090.** Hors périmètre : l'OBB (§3.1).

| Étage(s) | Config | map50 | map75 | map50-95 | P | R | F1 | zonemap | min/ép. |
|---|---|---|---|---|---|---|---|---|---|
| 0, 1 | 640 px, fl0, mo0, 50e | 0,880 | 0,816 | 0,650 | 0,806 | 0,841 | 0,822 | 49,3 | 0,28 |
| 1, 2 | **1280 px**, fl0, mo0, 50e | 0,890 | 0,861 | 0,701 | 0,859 | 0,871 | 0,864 | 41,3 | 0,72 |
| 1 | 1536 px, fl0, mo0, 50e | 0,875 | 0,825 | 0,670 | 0,829 | 0,845 | 0,836 | 42,1 | 1,08 |
| 2 | 1280, fl0, mo0,2 | 0,887 | 0,862 | 0,704 | 0,843 | 0,871 | 0,856 | 42,1 | 0,76 |
| 2 | 1280, fl0, mo1,0 | 0,894 | 0,846 | 0,682 | 0,819 | 0,864 | 0,840 | 45,7 | 0,82 |
| 2 | 1280, fl0,5, mo0 | 0,892 | 0,857 | 0,698 | 0,850 | 0,869 | 0,859 | 42,9 | 0,75 |
| 2, 3, 4 | 1280, **fl0,5, mo0,2**, 50e | 0,896 | 0,864 | 0,704 | 0,850 | 0,874 | 0,861 | 42,0 | 0,82 |
| 2 | 1280, fl0,5, mo1,0 | 0,889 | 0,838 | 0,674 | 0,813 | 0,859 | 0,835 | 45,3 | 0,82 |
| 3 | **yolo26n**, 1280, 50e | 0,877 | 0,825 | 0,664 | 0,802 | 0,848 | 0,823 | 50,7 | 0,58 |
| 4 | 1280, 30e | 0,886 | 0,845 | 0,677 | 0,828 | 0,860 | 0,843 | 45,6 | 0,78 |
| 4, 5 | 1280, **100e** — **det** | 0,888 | 0,860 | **0,706** | **0,866** | 0,869 | **0,867** | **38,8** | 0,79 |
| 5 | 1280, 100e — seg | 0,878 | 0,724 | 0,593 | 0,771 | 0,783 | 0,775 | 28,0 | 2,07 |

**Configuration retenue : `yolo26s`, détection, 1280 px à l'entraînement et à l'inférence,
100 époques.** Le run qui porte les chiffres ci-dessus est entraîné à `fliplr=0,5 mosaic=0,2` ; sur
les augmentations, voir la réserve de l'étage 2 — `fliplr=0 mosaic=0` est équivalent aux barres
d'erreur près et préférable par principe, mais **aucun run de 100 époques ne l'a encore mesuré**.

**Étage 1 — résolution : 1280 px, verdict franc.** 1280 ne gagne pas seulement en mAP (+0,051 sur
640) mais aussi au **rappel** (+0,030, le meilleur des trois), au `zonemap` (−8,0) et sur la
fragmentation — `n_split` 6,9 contre 10,9, `n_faux pos.` 3,5 contre 8,5. Les trois critères du
protocole concordent. **1536 est dominé sur toutes les métriques sans exception** : la résolution
aide, et elle sature entre 1280 et 1536.

*Nuance :* 1536 tourne à `batch=4` (carte de batch `640:16 1280:8 1536:4`, imposée par l'OOM CUDA).
Le batch *optimiseur* reste 64 dans les trois cas, vérifié dans l'ultralytics installé :
`accumulate = round(nbs/batch)` donne 4 / 8 / 16, `weight_decay` est remis à l'échelle par
`batch*accumulate/nbs` = 1, et `iterations = ceil(N/64)*epochs` ne dépend pas du batch — planning de
LR et nombre de pas d'optimisation sont donc identiques. Restent deux résidus défavorables au petit
batch, les statistiques de BatchNorm calculées sur le batch réel et la rampe de `warmup` qui
interpole `accumulate` de 1 à `nbs/batch`. Le batch ne variant **qu'à cet étage**, le biais est
confiné à la comparaison de résolutions ; un run de contrôle **1280 px à `batch=4`** (≈ 55 min) le
lèverait.

**Étage 2 — augmentations : seul `mosaic=1,0` est tranché.** Il coûte −0,019 à −0,027 de `map50-95`,
+4 de `zonemap` et −0,040 de précision : c'est le plus gros effet du balayage hors résolution et
backbone. En revanche les trois premières configurations tiennent dans **0,004 de `map50-95`**, et le
classement s'inverse selon la colonne : `fl0/mo0` a la meilleure précision, le meilleur F1 et le
meilleur `zonemap`, `fl0,5/mo0,2` le meilleur profil d'appariement en aire. **Ce balayage ne les
départage pas**, et les étages 3 à 5 ont donc été construits sur un gagnant arbitraire — sans
conséquence, puisque les trois configurations sont indiscernables. `fliplr=0 mosaic=0` est préférable
par principe (sémantiquement défendable, 12 % moins cher, meilleur F1 et meilleur `zonemap` à 50 e),
mais le run `100e @ fl0/mo0` manque pour le substituer à chiffres égaux.

Résultat accessoire mais instructif : que `fliplr=0,5` ne coûte **rien** de mesurable indique que le
modèle s'appuie sur la géométrie et la texture de la ligne, pas sur le sens de lecture.

**Étage 3 — backbone : `yolo26s`, dominé sur toutes les colonnes.** +0,040 de `map50-95`, +0,048 de
précision, `zonemap` −8,7, `n_split` −4,0 face à `yolo26n`, qui n'économise que 29 % de temps.

**Étage 4 — époques : 100, mais pas pour la raison affichée.** En mAP, 100e ne bat 50e que de 0,002,
soit du bruit. L'étage se tranche sur la décomposition : précision 0,866 contre 0,850, `zonemap`
**38,8 contre 42,0**, `n_faux pos.` **3,0 contre 4,9**, `n_split` **4,3 contre 7,6**. Signal
cohérent : `map50` *baisse* légèrement tandis que `map50-95` monte — les 50 époques supplémentaires
n'ajoutent pas de lignes trouvées, elles resserrent la localisation et suppriment les redondances,
pour 0,2 manque de plus par document. **100 n'est pas démontré comme le point de saturation** : le
mAP plateaure mais la décomposition s'améliorait encore, et la sélection par `best.pt` borne le
risque de sur-apprentissage. 150 et 200 valent le test.

**Étage 5 — formulation : det gagne la mesure, mais la mesure favorise det par construction.**
Trois faits vérifiés dans le code et les données :

- Les labels de `medieval-segmentation-yolo_lines` sont **exactement** la boîte englobante de ceux de
  `..._lines_polygon` — écart maximal 1e-6 en coordonnées normalisées, soit 0,00 px sur une page de
  4000 px, sur 3 440 lignes du split test. Mêmes classes, mêmes instances, mêmes splits : l'étage est
  proprement contrôlé, une seule variable change.
- `convert_lines_to_boxes()` réduit **les deux côtés** à la boîte englobante du `boundary`. Le modèle
  de détection est donc évalué sur précisément la grandeur qu'il optimise ; le modèle de segmentation
  produit des polygones ensuite écrasés en boîtes droites. Pour une ligne de 600×40 px inclinée de
  8°, cette boîte fait 600×123 px, soit **3,08× l'aire de la boîte tendue**.
- La signature de l'écart le confirme : **`map50` quasi identique (−0,010) et `map75` effondré
  (−0,136)**. Le modèle de segmentation trouve à peu près autant de lignes ; il perd à IoU serré,
  là où l'écrasement le pénalise. Les P/R/F1 étant pris à IoU 0,75, ils subissent le même biais.

Conformément au §1.4, le `zonemap` de 28,0 n'est **pas** comparé aux 38,8 de la détection. Il se lit
structurellement : il est **sous le plancher de 33,3** de tout modèle à boîtes, avec 0,106 d'aire de
faux positifs contre 0,333 pour l'oracle. Ce n'est pas « seg est meilleur » ; c'est « seg n'est pas
plafonné là où det l'est ». Et seg y laisse beaucoup : `match` 0,822 et `miss` 0,140 contre 0,969 et
0,025, `n_multiple` 84,0 contre 45,1 — un vrai problème d'appariement, donc une vraie marge. À
comparer à la bonne référence : ce run reproduit le profil ZoneMapAlt de `LINES-SEG` (28,0 contre
28,1) **tout en le dépassant largement en localisation** (`map75` 0,724 contre 0,428, F1 0,775 contre
0,555). C'est de loin le meilleur segmenteur mesuré sur ce corpus.

**L'arbitrage det vs seg n'est donc pas clos** — il attend la métrique baseline (§2.1) ou le CER
aval, seuls juges que le mAP de boîtes ne peut pas remplacer.

#### Temps observés, pour dimensionner la suite

| Config | durée | min/époque |
|---|---|---|
| 640 px / bs16 / det / 50e | 0 h 14 | 0,28 |
| **1280 px / bs8 / det** | 0 h 36 à 1 h 19 | **0,79** (0,72–0,82 sur 10 runs) |
| 1536 px / bs4 / det / 50e | 0 h 54 | 1,08 |
| 1280 px / bs8 / **seg** / 100e | 3 h 27 | 2,07 |

- **Le coût est linéaire en époques** : 0,78 min/ép. à 30e, 0,79 à 100e. Un run de 100e à 1280 px
  coûte 1 h 20, un run de 200e 2 h 40.
- **Échelle en résolution :** 640→1280 (×4 pixels) coûte ×2,57 ; 1280→1536 (×1,44 pixels) coûte
  ×1,50. La sous-linéarité à 640 px vient de ce que le GPU n'y est pas saturé ; à partir de 1280 px
  c'est linéaire en pixels.
- **La segmentation coûte ×2,60** à configuration égale : toute branche seg vaut 2,6 runs det.
- **`yolo26n` n'économise que 29 %** — trop peu pour servir de proxy exploratoire. Pour explorer
  large, baisser les époques (30e à 1280 px = 23 min) est un bien meilleur levier.

*Réserve sur `results/line_sweep_timings.csv` :* le fichier s'accumule sur plusieurs tentatives (le
run 640 px y figure 6 fois) et son schéma change en cours de route. Seul le dernier bloc est
cohérent avec les `min_par_epoque` de `summary.csv`. À corriger par un horodatage par ligne, ou un
écrasement par clé.

#### Le plancher de bruit n'est pas mesuré, et deux étages se sont joués dedans

Les étages 2 et 4 se décident à **0,002–0,004 de `map50-95`**, qu'un balayage mono-graine ne peut pas
résoudre. Indice que ce plancher n'est pas négligeable : la référence 640 px corrigée est à +0,004 de
`map50-95` du yolo26s 640 px figé mais **4,3 points de `zonemap` au-dessus** — les deux configs
diffèrent par les augmentations et la graine, donc c'est une borne supérieure du bruit, pas une
mesure. **Trois graines à la configuration retenue (≈ 4 h) sont le premier run à lancer**, avant tout
élargissement : sans elles, « 100e > 50e » et le choix d'augmentations ne sont pas établis.

---

## 2. Ce que l'instrument ne mesure pas encore

Trois manques, par ordre d'importance. Aucun n'invalide les chiffres du §1 ; tous limitent ce qu'on
peut en conclure.

### 2.1 La cible retenue n'est mesurée par rien

Le §1.5 l'a rendu concret : `convert_lines_to_boxes()` réduit référence et prédiction à la boîte
englobante du polygone. La comparaison est symétrique, mais **le mAP ne dit rien de la qualité des
polygones**, seulement de celle de leur boîte. Deux conséquences structurantes :

- **Le mAP ne peut pas arbitrer « détection vs segmentation vs OBB ».** Il n'est pas neutre : il
  mesure exactement l'objectif du détecteur de boîtes, et écrase ce que les deux autres formulations
  apportent. C'est la conclusion de l'étage 5.
- La **baseline**, cible retenue (décision 3), n'est mesurée par aucune métrique actuelle. Et une
  précision technique qui n'allait pas de soi : **le mAP repose sur l'IoU, donc sur des aires, et une
  baseline est une polyligne d'aire nulle.** Le mAP ne peut pas scorer des baselines.

La métrique de référence du domaine est celle du concours **cBAD** (Grüning et al.) : autour de
chaque baseline de référence, une zone de tolérance **de largeur proportionnelle à la hauteur locale
de la ligne** — une erreur de 5 px est grave sur une écriture de 20 px, négligeable sur une écriture
de 100 px. Une baseline prédite est appariée si elle reste dans cette zone sur une fraction
suffisante de sa longueur ; on en tire P/R/F1. Elle est invariante à la hauteur du polygone (qui ne
nous intéresse pas), capture nativement fusions et scissions, et rend nos chiffres comparables à la
littérature. **Environ 200 lignes à implémenter, et c'est un prérequis, pas un confort** — c'est
elle qui débloque l'arbitrage de formulation.

| Métrique | Ce qu'elle capture | Statut |
|---|---|---|
| mAP@50 / 75 / 50-95 | Qualité de la boîte englobante | Fiable, mais sensible aux confiances (§1.3) et aveugle au polygone |
| P/R/F1 @ IoU 0,75 | Taux réels au point d'opération | Fiable, même réserve de format |
| ZoneMapAlt | Fusions, scissions, manques | Fiable **au sein d'une même famille** (§1.4) |
| **P/R/F1 sur baselines (cBAD)** | **Ce qu'on veut réellement produire** | **À implémenter** |
| CER aval, MEDUSA fixé | Coût réel de la segmentation dans la pipeline | À calculer ponctuellement |
| Pages/h | Dimensionnement de l'objectif 2 | Dans `timings.csv`, à consigner |

Deux autres limites d'implémentation, en plus de la dissymétrie aire/comptage du §1.4 :

- **Une page sans prédiction sort du score au lieu d'être comptée en manques** : `_score_batch` fait
  `continue` avant le calcul ZoneMapAlt. Sans effet sur les modèles du §1.2, sensible dès qu'on
  filtre les prédictions.
- **L'agrégation inter-documents est une moyenne non pondérée**, donc un document d'une page pèse
  autant qu'un document de vingt.

Sur le CER aval : il ne s'agit pas d'en faire une métrique de segmentation, mais de le calculer deux
ou trois fois en fin de parcours, parce que métriques de segmentation et CER ne sont pas monotones
l'une dans l'autre :

| Erreur | Effet mAP | Effet CER |
|---|---|---|
| Polygone 8 px trop haut | Visible | Nul |
| Baseline décalée de 3 px | Faible | Nul |
| Deux lignes fusionnées | Faible | **Une ligne entière perdue** |
| Ligne marginale en trop | Fort | Faible |

### 2.2 Le corpus : propre, mais deux rôles à séparer

Les 286 pages d'ICDAR-CMMHWR sont un bon terrain de développement — quatre fois le test set interne
MEDUSA, donc quatre fois moins de variance d'échantillonnage. **Le recouvrement avec CATMuS est
nul**, vérifié ; les YOLO étant entraînés sur `medieval-segmentation-yolo`, d'origine CATMuS, les
chiffres du §1.2 ne sont pas contaminés.

Reste à séparer les deux rôles, sans quoi le seul corpus propre s'use par sur-ajustement de
sélection — et le §1.5 vient d'y consommer 12 évaluations :

- **dev set** — ICDAR-CMMHWR et/ou les pages CATMuS de validation, pour développer et sélectionner ;
- **test set gelé** — les 63 pages / 36 documents du test set interne MEDUSA, jamais utilisées en
  entraînement, ouvertes en fin de parcours pour le chiffre publiable.

Sur ces 63 pages, deux réserves : c'est peu, et le corpus a été construit pour *maximiser* la
diversité (13 langues, dates et types variés), donc la variance sera élevée — 2 points de mAP
d'écart peuvent n'être que du bruit. Reporter un **intervalle de confiance par bootstrap sur les
pages**, et inspecter à l'œil les 5 pires pages : sur 63, c'est faisable.

### 2.3 L'audit des annotations

`prepare-yolo-lines` est sain (`alto_lines_to_yolo.py:178-212` lit les dimensions réelles via PIL,
normalise en flottant à six décimales, clippe dans [0,1]), donc le soupçon d'erreurs d'annotation
porte sur les ALTO eux-mêmes, et il n'est pas levé.

**Mise en œuvre.** Un script d'audit indépendant de tout modèle, produisant un CSV d'indicateurs de
suspicion par page : image manquante ou de dimensions divergentes ; `TextLine` sans polygone, sans
baseline ou sans texte ; polygone dégénéré (aire nulle, moins de 3 points, auto-intersection) ; ligne
hors cadre ou hors de son `TextBlock` ; IoU entre lignes voisines au-delà d'un seuil ; hauteur
aberrante par rapport à la médiane de la page ; nombre de lignes hors de l'intervalle interquartile
du document ; incohérence de nomenclature Segmonto entre corpus. Puis tri par suspicion et
inspection visuelle du top 100. Le test le plus rentable reste cependant le plus simple :
**re-projeter les annotations sur les images et les regarder**, sur 50 pages échantillonnées
(`docworkflow print -t line` et les scripts de `outputs/phaseA_faux_positifs/scripts/`).

**Bénéfices.** Un corpus de test propre — sans quoi on mesure du bruit ; une estimation du **plancher
de performance atteignable** (si 8 % des lignes de test sont mal annotées, aucun modèle ne passera
au-dessus) ; et un contrôle qualité réutilisable sur les XML de production pour l'objectif 2.

**Difficultés.** Coût humain de la relecture, à contenir par échantillonnage. Et risque de
**circularité** : nettoyer par heuristiques revient à imposer un a priori sur ce qu'est une ligne. À
documenter, et à ne jamais appliquer au test set sans validation manuelle.

---

## 3. Construire le segmenteur

### 3.1 YOLO au propre : ce qui est acquis, ce qui reste

Le §1.5 a instruit l'essentiel de cette piste et fixé la configuration retenue. Ce qui suit est ce
qu'il reste à faire.

**Runs de confirmation à lancer**, par ordre de priorité et pour ≈ 16 h GPU au total — à comparer aux
10,7 h du balayage lui-même :

| # | Run(s) | Coût | Ce que ça résout |
|---|---|---|---|
| 1 | 3 graines à la config retenue | ≈ 4 h | Mesure le plancher de bruit. Les étages 2 et 4 se sont joués à ≤ 0,004 de mAP (§1.5) |
| 2 | 1280 px à `batch=4` | 55 min | Découple batch et résolution à l'étage 1 |
| 3 | 100e à `fl0/mo0` | 1 h 15 | Scelle l'étage 2, dont le gagnant est dans le bruit |
| 4 | 150e et 200e | 2 h + 2 h 40 | À 100e le mAP plateaure mais la décomposition s'améliorait encore |
| 5 | seg à 150–200e + critère polygone | 5–7 h | Rejoue l'étage 5 sur un critère qui ne désavantage pas seg par construction |

**Lecture visuelle à faire sur le run retenu.** Sur les mêmes pages, comparer les prédictions det et
seg en regardant spécifiquement les **lignes inclinées ou courbes** et les **lignes courtes** —
`InterlinearLine`, `DropCapitalLine`, `HeadingLine`, soit 8 657 instances sur 116 465 (7,4 % du
corpus), là où l'écrasement en boîte droite fait le plus de dégâts. L'hypothèse à falsifier est
simple : la segmentation suit la pente, la détection ne peut pas la représenter du tout
(`bbox_to_baseline` renvoie un segment strictement horizontal).

**Périmètre.** Aligner entraînement et inférence : si la production applique le modèle à des crops
de `MainZone`, l'entraînement doit se faire sur des crops de `MainZone`.

**Difficultés.** Lignes très courtes et très longues restent les cas d'échec attendus — prévoir un
breakdown par longueur. Et robustesse à des mises en page jamais vues, la BDD objectif n'étant pas
CATMuS.

#### L'OBB reste entièrement à faire

Les boîtes orientées sont le compromis naturel pour un objet de ratio 30:1 sur une page inclinée, et
elles ne sont utilisées nulle part : `grep -ri obb src/` ne renvoie **rien**. Quatre prérequis,
tous vérifiés dans le code, aucun bloquant mais aucun gratuit :

1. **Pas de dataset OBB** — `../data/data_yolo` ne contient que les boîtes droites
   (`..._lines`, 4 valeurs par ligne) et les polygones libres (`..._lines_polygon`, 8 à 32 valeurs).
   Bonne nouvelle : le dataset est **dérivable sans réannoter**, par rectangle de surface minimale sur
   les polygones — qui sont, on l'a vérifié au §1.5, la même annotation que les boîtes droites. C'est
   un script, pas une campagne d'annotation.
2. **`result.obb` n'est jamais lu** — `yolo_line.py:209` itère sur `result.boxes` et ne récupère les
   polygones que via `result.masks.xy`. Pour un modèle OBB, ultralytics laisse `result.boxes` à
   `None` : la boucle lèverait une `TypeError`, **avalée par le `except Exception` de
   `yolo_line.py:276-279`**, qui imprime la trace et continue. Le mode de défaillance n'est donc pas
   un plantage visible mais **des pages de sortie vides** avec un message noyé dans les logs.
3. **Les 4 coins n'atteignent jamais `polygon_to_baseline`** — `_yolo_box_to_line` prend bien une
   branche polygone dès que `mask_xy` est non vide, et cette branche gère ≥ 3 points ; mais `mask_xy`
   n'est alimenté que depuis `result.masks.xy`, nul pour un modèle OBB. Le seul chemin disponible est
   `bbox_to_baseline(x1,y1,x2,y2)`, qui renvoie un segment **horizontal** (mesuré : pente 0,00° sur un
   rectangle incliné de 8°). Le correctif est dans `_process_batch`, pas dans `_yolo_box_to_line` :
   lire `result.obb.xyxyxyxy` et passer ces 4 coins comme polygone.
4. **`polygon_to_baseline` ne gère pas correctement un quadrilatère** — son balayage vertical
   échantillonne `n_points` positions dont la première et la dernière tombent exactement sur `x_min`
   et `x_max`, où un rectangle incliné a une hauteur nulle : les deux extrémités de la baseline
   s'effondrent sur les coins. Mesuré sur un rectangle incliné de 8° : pente correcte de **7,98° sur
   les points internes**, mais **4,19° sur la baseline complète** — la pente est divisée par deux, et
   c'est invariant en `n_points` (identique à 8, 16, 32, 64), donc ce n'est pas un problème de
   résolution d'échantillonnage mais de placement des bornes. Sur un masque de segmentation réel
   l'effet est marginal ; sur un OBB à 4 coins il est **systématique**. Corriger en échantillonnant en
   retrait des bornes, ou en traitant le quadrilatère convexe à part. À couvrir par un test de
   régression.

### 3.2 La discipline des faux positifs

Traitable **sans réentraîner**, mais le périmètre a doublement rétréci. Le §1.4 a montré que l'écart
ZoneMapAlt avec Kraken est aux trois quarts un plancher de format, hors d'atteinte pour un détecteur
de boîtes ; et le modèle du §1.5 a absorbé l'essentiel du reste par le seul entraînement —
`n_faux pos.` 3,0 contre 9,2, `n_split` 4,3 contre 16,8.

1. **Réglage de la NMS.** Restait le levier principal sur les doublons ; son gain attendu a chuté
   d'autant. `ultralytics` expose `iou` et `agnostic_nms` à la prédiction ; sur des objets de ratio
   30:1 empilés verticalement, le seuil par défaut supprime mal deux boîtes qui se recouvrent
   partiellement en hauteur. **À balayer sur le nouveau modèle**, pas sur l'ancien : quelques minutes
   par point, et c'est la seule façon de savoir s'il reste quelque chose à prendre.
2. **Seuil de confiance.** 0,40 était le meilleur compromis sur le modèle figé (§1.4). **À
   recalibrer sur le modèle retenu**, dont le profil de faux positifs est tout autre. Et jamais
   utilisé pour départager deux modèles.
3. **Filtre géométrique.** Aucun faux positif n'atteint 10 lignes et leur médiane est à 1,46 ligne :
   **un tel filtre n'aurait presque rien à couper ici.** Il garde sa valeur comme garde-fou de
   contrôle qualité pour l'objectif 2, sur des mises en page inconnues, pas comme correctif.

### 3.3 Un candidat hors du paradigme YOLO — plan B

Kraken étant écarté de la production (décision 6), il ne reste qu'une seule famille implémentée.

| | Kraken (`blla`) | YOLO |
|---|---|---|
| Formulation | Segmentation **sémantique** dense : carte de pixels de baseline + séparateurs, puis composantes connexes | Détection **d'instances** : une boîte ou un polygone par ligne |
| Sortie native | **Baseline** suivant la courbure | Boîte ou polygone, dont il faut *extraire* une baseline |
| Objets de ratio 30:1 | Naturellement adapté (ni boîte, ni NMS) | Régime défavorable |
| Lignes courbes ou inclinées | Suivies | Approximées |
| Pénalité ZoneMapAlt de format | Nulle (polygone contre polygone) | **33,3 points de plancher** (§1.4) |

**Candidat concret : un modèle de baselines dense de type Doc-UFCN** (Teklia) — U-Net compact, conçu
pour les documents historiques, entraînable, installable par pip, **indépendant de Kraken**. Variante
utile : prédire deux cartes, les pixels de baseline *et* les séparateurs de début/fin, ce qui permet
de séparer deux colonnes adjacentes ou deux lignes serrées.

**Pourquoi c'est un plan B.** Le §1.5 a montré que la marge d'un YOLO bien réglé était large, et elle
est maintenant prise. L'argument qui subsiste, et qui reste bon, est que le paradigme dense **prédit
nativement la baseline** — la cible retenue — là où YOLO produit une aire dont il faut l'extraire.
À engager seulement si le §3.1 plafonne, c'est-à-dire si l'OBB et les runs de confirmation ne
rapportent plus rien.

**Difficulté.** Une nouvelle classe `BaseLine` à intégrer, donc un coût de développement réel, et un
modèle plus lent que YOLO. Acceptable compte tenu de la décision 9 (qualité d'abord).

À noter : même exclu de la production, **kraken-CATMuS mérite de rester dans le benchmark** comme
point de comparaison chiffré. Cela coûte une ligne de config.

### 3.4 De la baseline au crop : un réglage à ne pas négliger

Si la sortie finale est une baseline, il faut une règle déterministe pour en déduire la **hauteur du
crop** envoyé à MEDUSA. Or MEDUSA a été entraîné sur des crops de type CATMuS à marges étroites —
l'article applique même un effet `seam-crop edges` avec p = 1 dans le pipeline synthétique,
précisément pour imiter ces marges serrées.

**Une baseline parfaite avec une mauvaise politique de crop donnera un mauvais CER.** La règle
(hauteur estimée depuis l'interligne local ou la hauteur d'œil médiane de la page) tient en un ou
deux paramètres, se balaie en une journée, et peut valoir plusieurs points de CER.

Le benchmark en donne un argument empirique, et c'est ce qui le fait remonter dans les priorités.
`LINES-SEG` obtient 0,879 de `map50` et s'effondre à `map75` (0,428) : rapport 0,49, contre 0,92 pour
Kraken et 0,97 pour le modèle du §1.5. Il détecte les lignes aussi bien que les meilleurs, mais ses
polygones ne coïncident pas avec ceux de la référence — corroboré par 66 configurations
plusieurs-à-plusieurs par document, et par une aire de manques **et** de faux positifs simultanément
élevées. L'explication la plus probable n'est pas un défaut de qualité mais une **divergence de
convention** : ce qu'un modèle externe appelle « polygone de ligne » (hampes et jambages inclus,
marges plus larges) n'est pas ce que CATMuS appelle ainsi.

Autrement dit : **un simple écart de convention de polygone fait passer un modèle du premier au
dernier rang.** C'est exactement le risque que la politique de crop fait peser sur le CER en aval.

---

## 4. Le contexte perdu par le line-level

### 4.1 HTR multi-ligne

`VLMMultiLineHTR` n'étant qu'un squelette, c'est un développement à part entière. Quatre décisions de
conception :

- **Constitution du crop** — rectangle englobant *k* lignes (simple, mais embarque le bruit
  inter-lignes), ou composite des *k* crops **empilés verticalement** après recadrage individuel.
  La seconde reste plus proche de la distribution vue à l'entraînement et découple le multi-ligne de
  la qualité du layout ; probablement supérieure, et rarement essayée.
- **Format de sortie** — une ligne de texte par ligne d'image, séparateur par jeton dédié plutôt que
  `\n` (que le modèle produit spontanément), et un **repli quand le nombre de lignes retournées
  diffère de *k***, cas qui arrivera.
- **Budget de pixels** — MEDUSA est entraîné à `max_pixels = 512×28×28 = 401 408` ; empiler 4 lignes divise la résolution par ligne, sauf à augmenter le budget. **C'est l'arbitrage central : plus de contexte contre moins de pixels par ligne.** À mesurer, pas à supposer.
- **Fenêtre roulante** — fenêtre *k*, pas *s* < *k*, agrégation par vote au niveau caractère après alignement (ROVER) ou par log-probabilité. Le désaccord entre fenêtres est un **signal de risque gratuit**, utile au frontend de l'objectif 3.

**Bénéfices.** Le contexte visé — langue, abréviations, conventions — avec un effet attendu maximal
sur les langues peu représentées, vérifiable via la Figure 4 de l'article. Et un gain de débit
partiel : diviser par *k* le nombre de passes ne divise pas le coût par *k*, les jetons image
restant, mais amortit le prompt et le surcoût par appel.

**Difficultés.** **La raison de l'échec du page-level s'applique ici à moindre échelle** : plus la
sortie est longue, plus le modèle peut halluciner sur ses priors linguistiques. Le multi-ligne est un
curseur entre les deux régimes, pas une solution qualitativement différente — d'où l'importance de
**balayer *k* = 1, 2, 4, 8** et de voir où le CER se retourne. Cette courbe est un résultat en soi.
S'ajoutent le désalignement lignes/sorties, le coût de réentraînement (LoRA, sur 643 k lignes), et la
**réattribution des sorties aux lignes d'origine**, sans quoi l'évaluation n'est plus comparable aux
chiffres publiés.

### 4.2 Contexte à l'échelle du document

Les décisions 10 et 11 confirment l'homogénéité attendue — généralement une main et une langue — et
la disponibilité probable de la langue et de la cote. Trois voies, par coût croissant.

**Voie A — glossaire de document, deux passes, sans apprentissage.** Passe 1 en line-level standard ;
extraction des formes récurrentes fiables (vocabulaire, abréviations attestées, langue majoritaire) ;
passe 2 avec ce glossaire dans le prompt. **À faire en premier :** deux jours, aucun entraînement, et
cela **mesure directement s'il y a du signal à l'échelle du document** — si le gain est nul, on évite
d'engager des semaines sur B et C. Limite : coût doublé, acceptable pour une expérience de
faisabilité.

**Voie B — métadonnées explicites.** Le mécanisme existe : le slot `{conventions}` du prompt (Fig. 2
de l'article). L'extension ajoute des dimensions — langue, siècle, script, qualité — entraînées avec
ces slots remplis depuis les métadonnées du corpus source, et renseignées à l'inférence par le
manifeste IIIF, un classifieur page-level ou une première passe sur 10 lignes. Peu coûteux,
interprétable, utile au frontend. *Difficulté :* les métadonnées de la BDD seront incomplètes et
parfois fausses — mesurer la **robustesse à une métadonnée erronée** (« occitan » affirmé sur du
latin peut dégrader plus que le prompt neutre) et prévoir une valeur « inconnu » vue à
l'entraînement.

**Voie C — représentation latente de document.** Le **soft prompt appris au moment du test** optimise
par document quelques dizaines de vecteurs de préfixe sur ses propres pages, par auto-apprentissage
sur les prédictions confiantes de la passe 1 : peu de paramètres, aucun corpus supplémentaire, et le
ratio pages/document de la BDD objectif lui est très favorable. L'**encodeur de document appris de
bout en bout** est la version ambitieuse, et elle exige assez de documents *distincts* pour
généraliser — or on a 643 k lignes mais bien moins de documents.

*Difficultés communes :* **boucle de renforcement d'erreur** (le modèle s'adapte à ses propres
hallucinations), d'où l'auto-apprentissage restreint à la haute confiance ; coût d'inférence par
document ; et un **prérequis d'évaluation** — un test set organisé par document, que les 63 pages /
36 documents ne fournissent pas (≈ 1,75 page par document). **À anticiper.** Enfin, les recueils
multilingues existant même s'ils sont rares, le contexte document doit **proposer sans imposer** :
une dérogation par page, sinon un recueil bilingue se dégrade systématiquement.

### 4.3 Comparaison avec le page-level

Conservé comme point de comparaison (décision 12). ZoneMapAlt rend la comparaison directe **pour tout
système produisant des polygones** ; pour un VLM ne produisant que du texte, il reste l'alignement
global — concaténer la GT dans l'ordre de lecture et la sortie page-level, puis une distance
d'édition globale, comme les travaux cités dans l'article (Vidal et al. [26], le *character error
vector* [2]). Aucune coordonnée requise côté prédiction, mais sensible à l'ordre de lecture de la GT
et aveugle à la structure. **Un point de comparaison ponctuel — une journée — pas un chantier.**

---

## 5. Plan d'action

### Phase A — Construire le segmenteur

| Piste | Contenu | État |
|---|---|---|
| **A1. Balayage d'entraînement** | Résolution, augmentations, backbone, époques, det vs seg | **Fait** (§1.5) — config retenue : `yolo26s` det, 1280 px, 100e |
| **A2. Confirmations (priorité)** | 3 graines, contrôle `batch=4`, `100e @ fl0/mo0`, 150/200e | À lancer, ≈ 10 h GPU (§3.1) |
| **A3. OBB** | Dériver le dataset, lire `result.obb`, corriger les bornes de `polygon_to_baseline` | À faire, 4 prérequis identifiés (§3.1) |
| **A4. Métrique baseline (cBAD)** | ≈ 200 lignes — **débloque l'arbitrage de formulation** | À implémenter (§2.1) |
| B. Balayage NMS | Sur le **nouveau** modèle ; gain attendu réduit (§3.2) | À faire, une soirée |
| C. Politique de crop | baseline → hauteur ; balayage du paramètre ; **effet mesuré sur le CER** | À faire (§3.4) |
| D. Modèle dense (plan B) | Doc-UFCN ou équivalent — seulement si A plafonne | Différé (§3.3) |
| E. Robustesse | Les candidats sur 200 pages de la BDD objectif, inspection visuelle sans GT | À faire |
| F. Décision | mAP **à confiances comparables**, ZoneMapAlt **à famille comparable**, cBAD, CER aval, pages/h | Bloquée par A4 |

**Critère de sortie :** un segmenteur retenu, sans Kraken, avec débit mesuré. Feu vert de
l'objectif 2.

**Réglages acquis, à appliquer par défaut dès maintenant :** `restrict_to_layout: true` ;
`img_size` égal à la résolution d'entraînement, soit **1280 px** pour le modèle retenu ; `mosaic`
≤ 0,2 à l'entraînement, `fliplr` indifférent dans le bruit (§1.5, étage 2) ; **seuil de confiance à
recalibrer** sur le nouveau modèle (§3.2).

**Ce qui bloque encore une conclusion publiable :** la métrique baseline (A4) et l'audit des
annotations (§2.3). Tant qu'elles manquent, deux limites tiennent — la cible retenue n'est mesurée
par rien, et « détection vs segmentation vs OBB » ne peut pas être tranché, le mAP de boîtes mesurant
exactement l'objectif de l'une des trois formulations (§1.5, §2.1). Les pistes A2, A3 et B peuvent
avancer sans elle ; la ligne F ne peut pas conclure.

### Phase B — Contexte (pendant que `rtk` tourne)

| # | Action | Coût |
|---|---|---|
| 1 | Vérifier les métadonnées disponibles sur 20 manifestes de la BDD | 1 h |
| 2 | Glossaire de document, deux passes, sans entraînement (§4.2 A) | 2 j |
| 3 | Développer `VLMMultiLineHTR` + balayage *k* = 1, 2, 4, 8 | 1–2 sem. |
| 4 | Fenêtre roulante et consensus, si *k* > 1 gagne | 1 sem. |
| 5 | Métadonnées explicites en extension du slot `{conventions}` | Réentraînement |
| 6 | Soft prompt au niveau document, conditionné aux résultats de 2 et 5 | Ouvert |

### À éviter

- **Conclure « détection vs segmentation » sur le mAP de boîtes.** Les labels de détection sont
  exactement la boîte englobante des polygones, et la métrique compare des boîtes englobantes : elle
  mesure l'objectif du détecteur et écrase ce que la segmentation apporte (§1.5, étage 5).
- **Comparer un `zonemap/score` de détection à un `zonemap/score` de segmentation.** 33,3 des points
  d'un détecteur de boîtes sont un plancher de format (§1.4). Comparer au sein d'une famille, ou sur
  les termes qui n'en dépendent pas (fusions, scissions, manques).
- **Trancher un étage du balayage sur un écart de ≤ 0,005 de `map50-95`** sans plancher de bruit
  mesuré. C'est le cas des étages 2 et 4 (§1.5).
- **Monter `img_size` à la prédiction seule.** Mesuré : jusqu'à −41 % de `map50-95` (§1.5).
- **Reconduire le seuil de confiance 0,40 tel quel.** Calibré sur un modèle à `n_FA` = 9,2 ; le modèle
  retenu est à 3,0 (§3.2).
- **Sélectionner un modèle sur une seule métrique.** Le classement s'inverse selon la métrique
  retenue ; c'est précisément ce qu'il faut regarder (§1.3).
- **Lire `map50` / `map75` entre YOLO et Kraken comme une comparaison géométrique.** Six points
  viennent des confiances, que Kraken ne produit pas (§1.3).
- **Diviser `zonemap/false_alarm` par `zonemap/n_false_alarm`.** Les deux ne portent pas sur la même
  population (§1.4).
- **Brancher un modèle OBB sur `YoloLineTask` en l'état.** Le `except Exception` de `_process_batch`
  avale la `TypeError` et produit des pages vides sans erreur visible (§3.1).
- **Relancer le page-level maintenant.** L'article a tranché et documenté pourquoi (§6 : les
  hallucinations « prevent us from moving on to the page level »). Le multi-ligne est la façon
  contrôlée d'explorer cet axe.

---

## 6. Trois résultats à viser

1. **Le coût chiffré de la segmentation dans une pipeline HTR VLM** : écart de CER entre segmentation
   de référence et segmentation automatique, à modèle HTR constant. Rarement mesuré proprement, et
   directement utile à la communauté.
2. **La courbe contexte / hallucination en fonction du nombre de lignes**, qui donne une réponse
   quantitative au débat page-level vs line-level ouvert dans l'article.
3. **Une méthode d'adaptation au niveau document** pour l'HTR historique, si les voies A puis C
   confirment qu'il y a du signal.

---

## 7. Décisions qui restent ouvertes

1. **Corpus d'entraînement du segmenteur** : CATMuS seul, ou CATMuS + les 142 000 lignes produites
   pour MEDUSA ? La seconde option élargit la couverture de mises en page mais mélange des
   conventions de layout — et il faut vérifier qu'aucune de ces pages ne recoupe le test set interne.
2. **Corpus d'évaluation** : ICDAR-CMMHWR comme dev set et les 63 pages MEDUSA gelées comme test, ou
   l'inverse ? Le §2.2 penche pour la première option, et le balayage vient d'y consommer 12
   évaluations : la formaliser devient urgent.
3. **Intégration d'un second paradigme** (§3.3) : le coût de développement d'un modèle dense est-il
   acceptable ? La question n'est plus « YOLO est-il assez bon ? » — il est maintenant en tête du
   benchmark — mais « veut-on un modèle qui prédise la baseline nativement plutôt que de l'extraire
   d'un polygone ? ». À trancher après l'OBB (A3) et la métrique baseline (A4).
4. **Traitement des doublons** : réglage de la NMS, ou acquis par l'entraînement ? Le §1.5 en a
   supprimé les deux tiers sans y toucher ; le balayage NMS sur le nouveau modèle dira s'il reste
   quelque chose à prendre.
5. **Quand implémenter la métrique baseline (cBAD)** : c'est le seul verrou qui bloque à la fois
   l'arbitrage « det vs seg vs OBB » et toute conclusion publiable (§2.1). Le balayage a produit ses
   candidats ; l'échéance est donc atteinte.
