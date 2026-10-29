# Darija GPT Lab

Construire un LLM from scratch pour la darija marocaine.

La progression pédagogique suit les concepts de [LLMs-from-scratch de Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch).

Les données proviennent des datasets publiés par [AtlasIA](https://huggingface.co/atlasia). Le vocabulaire et les fusions BPE sont appris sur notre corpus d'entraînement. Le projet n'utilise actuellement ni tokenizer préentraîné ni poids de modèle préentraîné.

## État actuel

Le chapitre 2 est en cours : tokenisation, normalisation Unicode, construction du vocabulaire, BPE et préparation des fenêtres glissantes. Le tokenizer est opérationnel et vérifié sur des exemples locaux. Le réseau GPT et son entraînement seront ajoutés au fil de l'étude des chapitres suivants.

## Contenu

- `scripts/atelier_tokenizer.py` : tokenizer simple par mots, apprentissage BPE, sauvegarde et préparation des paires entrée/cible.
- `docs/cours_creer_tokenizer_darija.pdf` : cours en français de 13 pages avec exemples et exercices corrigés.
- `tests/test_tokenizer.py` : vérifications du décodage Unicode, des tokens spéciaux, de la sauvegarde et des fenêtres glissantes.
- `requirements.txt` : version de la bibliothèque utilisée pour l'atelier.

## Installation

Python 3.10 ou plus récent est nécessaire. Depuis le dossier du dépôt :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Lancer l'atelier

```powershell
python scripts/atelier_tokenizer.py --limit 1000 --vocab-size 2000
```

Lors de sa première exécution, le script télécharge le fichier public `atlasia_stories.csv` de [atlasia/darija_english](https://huggingface.co/datasets/atlasia/darija_english) et utilise uniquement sa colonne `darija`.

Pour utiliser un CSV déjà téléchargé :

```powershell
python scripts/atelier_tokenizer.py --csv data/atlasia_stories.csv --limit 1000
```

Le script normalise en NFC, retire les doublons exacts, sépare les documents en entraînement et validation, puis apprend le tokenizer uniquement sur l'entraînement. Les documents apparentés demanderaient un regroupement supplémentaire pour une évaluation approfondie.

Les résultats sont écrits dans `atelier_darija/` :

- `tokenizer_darija.json` : vocabulaire, fusions et configuration du tokenizer ;
- `manifest.json` : source du CSV, hash SHA-256, configuration et mesures sur la validation.

Les fichiers téléchargés et les résultats générés sont exclus de Git.

## Recharger le tokenizer

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("atelier_darija/tokenizer_darija.json")
tokenizer.encode_special_tokens = True

text = "salam, labas?"
encoded = tokenizer.encode(text, add_special_tokens=False)
print(encoded.ids)
print(tokenizer.decode(encoded.ids, skip_special_tokens=False))
```

Avec Tokenizers 0.22.1, le drapeau `encode_special_tokens=True` doit être rétabli après rechargement. Il garde les chaînes littérales comme `[EOS]` dans le texte ordinaire. Le véritable marqueur de fin de document est ajouté explicitement par son identifiant.

## Vérifications

```powershell
python -m unittest discover -s tests -v
```

Les tests utilisent des exemples inventés, identifiés comme tels. Ils ne mesurent pas la qualité du tokenizer sur le corpus AtlasIA. Le téléchargement réseau du CSV est distinct de ces tests.

## Données et attribution

Le dataset `atlasia/darija_english` affiche une licence **CC BY-NC 4.0** sur sa [page Hugging Face](https://huggingface.co/datasets/atlasia/darija_english). Les conditions et attributions des données restent applicables indépendamment du code. Les fichiers du corpus ne sont pas redistribués dans ce dépôt.

[Atlaset](https://huggingface.co/datasets/atlasia/Atlaset) est une autre ressource possible pour la suite. Son accès nécessite d'accepter les conditions sur Hugging Face ; il n'est pas téléchargé par ce script.

Le cours et le code de cet atelier ont été préparés pour notre apprentissage. Le dépôt du livre reste la référence pour la progression pédagogique ; ses notebooks et ses poids ne sont pas copiés ici.

## Progression prévue

1. Préparer et inspecter un corpus darija.
2. Construire le tokenizer et les fenêtres glissantes.
3. Implémenter l'attention et les blocs Transformer.
4. Assembler un petit GPT et apprendre ses paramètres à partir de zéro.
5. Évaluer les générations, puis étudier l'adaptation aux instructions.
