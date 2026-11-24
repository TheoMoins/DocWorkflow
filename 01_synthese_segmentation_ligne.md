# Synthèse — Segmentation de ligne et production des XML

> Document de synthèse. Il reprend **strictement** le contenu des notes, réorganisé et clarifié,
> sans ajout d'analyse ni de recommandation (voir document 2 pour cela).

---

## 0. Cadre général

Trois objectifs, dans cet ordre chronologique :

1. **Reprendre la partie sur la segmentation de ligne.**
2. Une fois la pipeline complète jugée suffisamment satisfaisante, **appliquer `rtk` sur la BDD
   objectif** : plusieurs centaines de manifestes, des centaines de milliers de pages à produire
   en XML.
3. **Réaliser une application** (ou du moins un frontend) permettant à n'importe quel utilisateur
   de charger une ou plusieurs pages et d'en demander la transcription, hébergée sur le serveur du
   laboratoire qui contient les modèles et les GPUs.

Deux dépôts git : `docworkflow` (principal) et `rtk` (extraction et passage à l'échelle).
Les avancées récentes ont porté sur la partie HTR, avec le développement de **MEDUSA**, dont
l'article a été accepté en conférence. Les résultats sur cette partie sont satisfaisants ; le
présent document porte sur ce qui reste à faire, **à savoir l'objectif 1 : la segmentation de
ligne**.

---

## 1. Deux stratégies possibles pour produire les fichiers XML

| # | Stratégie | Description |
|---|-----------|-------------|
| 1 | **HTR page-level** | Transcription directe de la page entière, sans étape de segmentation |
| 2 | **Segmentation de ligne + HTR line-level** | Détection des lignes, puis transcription ligne à ligne |

---

## 2. Bilan de la stratégie 1 (page-level)

Cette option a été **partiellement explorée**, en essayant d'adapter MEDUSA au niveau page, **sans
succès**. Deux causes ont été identifiées.

### 2.1 Hétérogénéité du contenu entre pages

La différence de contenu d'une page à l'autre est forte : certaines pages ne contiennent que
quelques lignes, d'autres jusqu'à une centaine. Cela rend **difficile voire impossible
l'optimisation de la fenêtre à considérer**, et crée des problèmes.

### 2.2 Impossibilité d'une comparaison honnête en CER

Comparer line-level et page-level en CER n'est pas honnête en l'état, à cause de :

- la difficulté de **scorer les lignes dans le bon ordre** ;
- la nécessité de **prendre en compte les duplications ou suppressions de lignes** potentielles ;
- etc.

C'est ce qui a motivé l'implémentation de **ZoneMapAlt**. Mais cette métrique **présuppose de
disposer des polygones de lignes** : il n'est donc pas possible de comparer avec un VLM qui
fournirait simplement un texte à l'échelle de la page, sans segmentation des lignes.

### 2.3 Orientation qui en découle

Cela conduit à penser qu'il faut **se tourner vers des méthodes qui fournissent le résultat ligne
par ligne avec coordonnées**, afin de faciliter la comparaison et d'éviter le problème de
l'ordonnancement des lignes. *Cette conclusion reste toutefois formulée avec incertitude dans les
notes.*

---

## 3. État actuel de la segmentation de ligne

- **Ce qui fonctionne le mieux semble être Kraken.**
- Il faudrait néanmoins **refaire le benchmark au propre**, ainsi qu'un **entraînement d'un YOLO au
  propre, avec des données clean**.
- De mémoire, une partie des difficultés provenait aussi d'**erreurs dans l'annotation des
  données**.
- En tout état de cause, **`YoloLine` (en segmentation comme en détection) n'a jamais donné de
  résultats très satisfaisants** — ce qui est jugé **surprenant**.
- **La solution la moins ambitieuse à court terme** serait de développer enfin un **modèle de
  segmentation de ligne qui soit efficace**.

---

## 4. Défauts de la stratégie 2 (line-level)

### 4.1 Temps de calcul

Potentiellement assez long. YOLO fait l'étape relativement rapidement tout compte fait, mais
**n'est pas optimal pour l'instant en termes de performance**.

### 4.2 Absence de contexte

La transcription ligne à ligne **ne bénéficie pas du contexte des lignes précédentes et
suivantes**, alors que celui-ci pourrait apporter une vraie information, notamment :

- la compréhension de **la langue dans laquelle on se situe** ;
- les **abréviations** ;
- les **conventions d'écriture** ;
- etc.

---

## 5. Pistes envisagées

### 5.1 HTR multi-ligne

Idée : traiter **plusieurs lignes à la fois**, afin de :

- **conserver un contrôle sur l'output en sortie** (étant entendu que la ligne est déjà une unité
  très hétérogène) ;
- **prendre en compte un contexte plus large**.

Extension possible : un **algorithme à fenêtre roulante**, afin de **croiser les résultats** et de
**s'assurer de la stabilité de la prédiction**.

### 5.2 Contexte à l'échelle du document

Prolongement naturel de l'idée précédente. D'une page à l'autre **au sein d'un même document**,
beaucoup d'informations sont partagées :

- qualité du document ;
- script ;
- langue ;
- conventions ;
- etc.

La question ouverte est **comment bénéficier de ce partage d'information**. Deux voies envisagées :

1. une **étape d'apprentissage de métadonnées en dur** ;
2. un **embedding latent**, en quelque sorte une **représentation de document qui apprendrait
   elle-même les dimensions à considérer**.

Cette seconde voie est identifiée comme **peut-être ambitieuse**, et posant la question du
**volume de données nécessaire**.

---

## 6. Récapitulatif des points ouverts tels qu'énoncés

| Point | Statut dans les notes |
|-------|----------------------|
| Adaptation de MEDUSA au page-level | Exploré partiellement, sans succès |
| Comparaison honnête CER page/line | Non résolue ; ZoneMapAlt implémenté mais exige les polygones |
| Choix de méthodes « ligne + coordonnées » | Orientation privilégiée, mais **incertitude assumée** |
| Kraken | Meilleure solution actuelle, à re-benchmarker au propre |
| YOLO ligne (détection et segmentation) | Résultats jamais satisfaisants, jugé surprenant |
| Qualité des annotations | Suspectée comme source d'erreurs |
| Temps de calcul du line-level | Identifié comme défaut potentiel |
| Multi-ligne + fenêtre roulante | Idée à explorer |
| Contexte document (métadonnées vs embedding latent) | Idée à explorer, ambition et volumétrie à évaluer |

---

## Annexe — Décisions arrêtées (réponses aux questions de clarification)

Les questions posées à la lecture des notes ont été tranchées. Réponses consignées ici ;
conséquences développées dans le document 3.

| # | Question | Décision |
|---|----------|----------|
| 1 | Corpus de référence du benchmark | **Test set interne MEDUSA** (63 pages / 36 documents), jamais utilisé en entraînement |
| 2 | Entrée de l'étape ligne | Pipeline **de bout en bout**. Le layout n'est pas indispensable, mais utile pour (a) ne conserver que la `MainZone`, (b) ordonner les lignes zone par zone si besoin |
| 3 | Cible prédite | La **baseline** in fine. Si un polygone est disponible, la baseline s'en déduit. Les deux formats restent envisageables |
| 4 | Critère de succès | **mAP** privilégié. Métriques orientées baseline à découvrir. Le CER n'est pas jugé pertinent comme métrique de segmentation |
| 5 | Configuration YOLO déjà testée | Non mémorisée — motif principal du re-benchmark au propre |
| 6 | Kraken | Modèle entraîné sur CATMuS. **Pas de fine-tuning, et objectif d'éviter Kraken** en production |
| 7 | ZoneMapAlt | Implémentée et fonctionnelle |
| 8 | `VLMMultiLineHTR` | **Squelette uniquement**, jamais développé |
| 9 | Arbitrage qualité / débit | **Qualité d'abord.** Le temps de calcul ne sera troqué qu'ensuite, si nécessaire |
| 10 | Homogénéité des documents | Recueils possibles avec plusieurs textes, mais **généralement une main et une langue**, avec exceptions |
| 11 | Métadonnées disponibles | Probablement **langue et cote** — à vérifier |
| 12 | Page-level | **Conservé comme point de comparaison** |
