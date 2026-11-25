# Darija LLM Lab — Model Card

## Model summary

Darija LLM Lab is an educational proof of concept for a small Moroccan Darija language model built from randomly initialized weights. The notebooks train a ByteLevel BPE tokenizer and a decoder-only GPT using next-token prediction. This repository is a learning project, not a production-ready assistant.

## Architecture

The baseline configuration uses a **2,000-token vocabulary**, **64-token context**, **64-dimensional embeddings**, **2 Transformer blocks**, and **4 causal attention heads**. Token and learned position embeddings feed the decoder; a linear output head predicts the next token at each position.

### Overall model

![Darija GPT architecture: embeddings, Transformer blocks, and next-token output](assets/diagrams/08-darija-gpt-architecture.png)

### Decoder block

![Transformer decoder block with causal attention and feed-forward network](assets/diagrams/07-transformer-block.png)

Each block applies causal self-attention and a position-wise feed-forward network along normalized residual paths.

### Multi-head attention

![Multi-head causal attention with parallel heads and combined output](assets/diagrams/06-multi-head-attention.png)

The attention heads process different learned feature subspaces in parallel and combine their outputs.

## Intended use

- Learning how tokenization, causal attention, Transformer blocks, and next-token training work.
- Small-scale experiments with Moroccan Darija text.
- Inspecting training behavior and text generation as a proof of concept.

Do not use the model as a dependable translation, moderation, legal, medical, or factual-answering system.

## Training data

The notebooks use the `darija` column from the [AtlasIA `darija_english` dataset](https://huggingface.co/datasets/atlasia/darija_english). Documents are normalized, deduplicated, and split by document into training, validation, and test sets. The tokenizer is fit on training text only; validation is used for checkpoint selection, and the test split is reserved for final evaluation. Dataset terms and attribution requirements are described on the dataset's Hugging Face page.

## Training procedure

The GPT is initialized from random weights and trained with next-token cross-entropy. The notebooks use AdamW, measure validation loss, and retain the checkpoint with the best validation result. Later notebooks resume training or compare a scheduled learning-rate experiment. These settings describe the current small educational setup and may change as the project evolves.

## Limitations

- The model and corpus are small; generated text can be repetitive, incoherent, or factually wrong.
- Darija spelling, dialect, and code-switching vary widely and may be unevenly represented in the source data.
- Exact duplicate documents are removed, but near duplicates may remain across splits.
- Evaluation results are proof-of-concept measurements and do not establish broad language quality or safety.
- The dataset's non-commercial terms apply to the data independently of this code; review the source license before any reuse.

## Artifacts and evaluation

Model weights, tokenizer files, datasets, and evaluation artifacts are generated locally by the notebooks and are excluded from this Git repository. The repository therefore documents the model implementation and training workflow; it does not distribute a ready-to-download checkpoint.

For the reproducible workflow and notebook index, see the [project README](README.md). The notebooks report the actual run metrics when executed.
