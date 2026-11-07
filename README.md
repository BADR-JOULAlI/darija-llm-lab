# Darija LLM Lab

Projet pédagogique en Python et PyTorch pour construire un LLM from scratch dédié à la darija marocaine : préparation des données AtlasIA, création d’un tokenizer BPE, implémentation de l’attention et des blocs Transformer, préentraînement à partir de poids aléatoires et évaluation de la génération de texte.

La progression pédagogique suit les concepts de [LLMs-from-scratch de Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch).

Les données proviennent des datasets publiés par [AtlasIA](https://huggingface.co/atlasia). Le vocabulaire et les fusions BPE sont appris sur notre corpus d'entraînement. Le projet n'utilise actuellement ni tokenizer préentraîné ni poids de modèle préentraîné.

## État actuel

Le projet est organisé en **notebooks Jupyter autonomes**, avec des explications en français, du code cellule par cellule et des exercices corrigés. Les notebooks 01 à 03 étudient les données, le tokenizer et les fenêtres glissantes. Le notebook 04 commence le chapitre sur l'attention causale.

Les notebooks sont livrés **sans exécution ni sorties enregistrées**. Aucun téléchargement ni apprentissage n'est lancé lors de leur création. Le tokenizer sera effectivement appris et sauvegardé quand la cellule d'entraînement du notebook 02 sera exécutée.

## Notebooks à suivre dans l'ordre

| Notebook | Ce qu'on construit |
| --- | --- |
| [01 - Préparer les données darija](notebooks/01_preparer_donnees_darija.ipynb) | Lecture de la colonne `darija`, normalisation NFC, déduplication, échantillonnage et séparation entraînement/validation |
| [02 - Créer notre tokenizer](notebooks/02_creer_tokenizer_darija.ipynb) | Tokenizer par mots, compréhension de BPE, apprentissage ByteLevel BPE, vérifications, mesures et sauvegarde |
| [03 - Fenêtres glissantes](notebooks/03_fenetres_glissantes.ipynb) | Entrées/cibles décalées, longueur de contexte, stride, fin de document et petits lots |
| [04 - Attention causale](notebooks/04_attention_causale.ipynb) | Requêtes, clés, valeurs, poids d'attention, masque causal et vecteurs de contexte avec de petits calculs Python |

Le code utile est directement dans les notebooks ; aucun script Python externe au projet n'est nécessaire pour les suivre.

## Installation

Python 3.10 ou plus récent est nécessaire. Depuis le dossier du dépôt :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Ouvrir le projet

```powershell
python -m jupyterlab notebooks
```

Ouvre le notebook 01, sélectionne le noyau Python de l'environnement installé, puis avance cellule par cellule. Le même projet peut être ouvert dans VS Code avec la prise en charge Jupyter.

## Données locales et résultats

Pour commencer, le notebook 01 attend `data/raw/atlasia_stories.csv`, issu de [atlasia/darija_english](https://huggingface.co/datasets/atlasia/darija_english), et utilise uniquement sa colonne `darija`.

Tu peux copier un CSV déjà téléchargé dans ce dossier. Le téléchargement intégré est **désactivé par défaut** ; il demande de passer explicitement `DOWNLOAD_DATA = True` dans le notebook 01. Les fichiers existants sont réutilisés.

Les fichiers produits lors de l'exécution sont :

- `data/processed/train.jsonl` et `validation.jsonl` : partitions par documents ;
- `data/processed/corpus_manifest.json` : provenance, hashes et préparation du corpus ;
- `artifacts/tokenizer/tokenizer_darija.json` : tokenizer appris dans le notebook 02 ;
- `artifacts/tokenizer/training_manifest.json` et `evaluation.json` : paramètres d'apprentissage, provenance et mesures de validation.

Les dossiers `data/` et `artifacts/` sont exclus de Git. Les documents apparentés et les doublons proches demanderaient une analyse supplémentaire avant une évaluation approfondie.

## Recharger le tokenizer

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("artifacts/tokenizer/tokenizer_darija.json")
tokenizer.encode_special_tokens = True

text = "salam, labas?"
encoded = tokenizer.encode(text, add_special_tokens=False)
print(encoded.ids)
print(tokenizer.decode(encoded.ids, skip_special_tokens=False))
```

Avec Tokenizers 0.22.1, le drapeau `encode_special_tokens=True` doit être rétabli après rechargement. Il garde les chaînes littérales comme `[EOS]` dans le texte ordinaire. Le véritable marqueur de fin de document est ajouté explicitement par son identifiant.

## Vérifications dans les notebooks

Les assertions sont placées à côté des notions étudiées : partitions sans doublons exacts communs, décodage après normalisation Unicode, chaînes spéciales littérales, stabilité après rechargement du JSON, décalage entrée/cible et frontières des documents. Les exemples numériques ou inventés sont identifiés comme pédagogiques.

Ces vérifications ne mesurent pas la qualité d'un futur LLM. La validation du tokenizer doit être faite sur les documents réservés, sans les utiliser pour apprendre le vocabulaire ou les fusions.

## Données et attribution

Le dataset `atlasia/darija_english` affiche une licence **CC BY-NC 4.0** sur sa [page Hugging Face](https://huggingface.co/datasets/atlasia/darija_english). Les conditions et attributions des données restent applicables indépendamment du code. Les fichiers du corpus ne sont pas redistribués dans ce dépôt.

[Atlaset](https://huggingface.co/datasets/atlasia/Atlaset) est une autre ressource possible pour la suite. Son accès nécessite d'accepter les conditions sur Hugging Face ; les notebooks actuels utilisent le dataset public `darija_english`.

Les explications et le code de ces notebooks ont été préparés pour notre apprentissage. Le dépôt du livre reste la référence pour la progression pédagogique ; ses notebooks et ses poids ne sont pas copiés ici.

## Progression prévue

1. Préparer et inspecter un corpus darija.
2. Construire le tokenizer et les fenêtres glissantes.
3. Comprendre l'attention causale, puis l'implémenter avec PyTorch et ajouter le multi-têtes.
4. Assembler un petit GPT et apprendre ses paramètres à partir de zéro.
5. Évaluer les générations, puis étudier l'adaptation aux instructions.
