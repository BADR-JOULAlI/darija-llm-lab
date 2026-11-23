# Darija LLM Lab

An educational Python and PyTorch project for building a Darija language model from scratch: preparing AtlasIA data, training a BPE tokenizer, implementing attention and Transformer blocks, pretraining from random weights, and evaluating text generation.

The notebooks build the project step by step, from Darija text preparation and tokenization to a Transformer language model. All project code and examples are written for Moroccan Darija.

Data comes from datasets published by [AtlasIA](https://huggingface.co/atlasia). The vocabulary and BPE merges are learned from our training corpus. The project currently uses neither a pretrained tokenizer nor pretrained model weights.

## Current status

The project is organized as **standalone Jupyter notebooks** with explanations and code cells. Notebooks 01–03 prepare the data, tokenizer, and sliding windows. Notebooks 04–06 introduce causal attention, implement it in PyTorch, and extend it to multiple attention heads. Notebook 07 assembles a Transformer block. Notebook 08 puts the components together into a GPT, trains it on the Darija corpus, and evaluates it on held-out data. Notebook 09 reviews the saved metrics and generated samples.

All nine notebooks have been run from top to bottom in order, and their outputs are saved in the notebooks so results are visible on GitHub. Start with notebook 01: it downloads the pinned CSV if needed, notebook 02 trains the tokenizer, notebook 08 trains a small GPT, and notebook 09 reads its evaluation artifacts. The saved model, tokenizer, and dataset files remain local and are excluded from Git.

## Notebooks

| Notebook | What it builds |
| --- | --- |
| [01 - Prepare Darija data](notebooks/01_prepare_darija_data.ipynb) | Read the `darija` column, NFC normalization, deduplication, sampling, and train/validation splits |
| [02 - Build our tokenizer](notebooks/02_build_darija_tokenizer.ipynb) | Word-tokenizer example, BPE concepts, ByteLevel BPE training, checks, measurements, and saving |
| [03 - Sliding windows](notebooks/03_sliding_windows.ipynb) | Shifted inputs and targets, context length, stride, document boundaries, and batches |
| [04 - Causal attention concepts](notebooks/04_causal_attention_concepts.ipynb) | Queries, keys, values, attention weights, causal masking, and context vectors with small Python calculations |
| [05 - Causal attention in PyTorch](notebooks/05_causal_attention_pytorch.ipynb) | Trainable Q/K/V projections, causal masking, batches, and a reusable module |
| [06 - Multi-head attention](notebooks/06_multi_head_attention.ipynb) | Split Q/K/V into heads, compute masked attention in parallel, combine head outputs, and verify tensor shapes |
| [07 - Transformer block](notebooks/07_transformer_block.ipynb) | Pre-normalization, residual connections, multi-head causal attention, and a GELU feed-forward network |
| [08 - Train and evaluate a Darija GPT](notebooks/08_train_and_evaluate_darija_gpt.ipynb) | Token and position embeddings, class-based GPT model, document-safe batches, training loop, validation loss, perplexity, checkpointing, and text generation |
| [09 - Review model evaluation](notebooks/09_review_model_evaluation.ipynb) | Load saved metrics, inspect validation results and generated samples, and calculate simple n-gram diversity indicators |

## Install

Python 3.10 or newer is required. From the repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Open the project

```powershell
python -m jupyterlab notebooks
```

Open notebook 01, select the Python environment installed above, and use **Run All**. Continue with notebooks 02 through 09 in order; each notebook starts a fresh kernel, so run the preceding notebooks first to create its required files. Notebook 01 needs an internet connection the first time it downloads the pinned CSV. Notebook 08 performs a 200-step training run and can take longer on a CPU. The project can also be opened in VS Code with Jupyter support.

## Local data and outputs

Notebook 01 uses only the `darija` column from [atlasia/darija_english](https://huggingface.co/datasets/atlasia/darija_english).

Notebook 01 downloads the pinned CSV into this directory when it is missing. You can also place a CSV here and set `DOWNLOAD_DATA = False` in the notebook. Existing files are reused.

Files produced when you run the notebooks:

- `data/processed/train.jsonl` and `validation.jsonl`: document splits;
- `data/processed/corpus_manifest.json`: data provenance, hashes, and preprocessing settings;
- `artifacts/tokenizer/tokenizer_darija.json`: tokenizer trained in notebook 02;
- `artifacts/tokenizer/training_manifest.json` and `evaluation.json`: training settings, provenance, and validation measurements.

The `data/` and `artifacts/` directories are excluded from Git. Related documents and near-duplicates need further analysis before a rigorous evaluation.

## Reload the tokenizer

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("artifacts/tokenizer/tokenizer_darija.json")
tokenizer.encode_special_tokens = True

text = "salam, labas?"
encoded = tokenizer.encode(text, add_special_tokens=False)
print(encoded.ids)
print(tokenizer.decode(encoded.ids, skip_special_tokens=False))
```

With Tokenizers 0.22.1, restore the `encode_special_tokens=True` flag after reloading. It keeps literal strings such as `[EOS]` as ordinary text. The real document-ending marker is added explicitly by its ID.

## Notebook checks

Assertions are placed next to the concepts they check: splits have no exact duplicates in common, Unicode text round-trips after normalization, literal special-token strings are handled correctly, saved tokenizer IDs remain stable after reloading, targets are shifted by one token, and windows preserve document boundaries.

These checks do not measure the quality of a future LLM. The tokenizer must be evaluated on reserved documents that were not used to learn its vocabulary or merges.

## Data and attribution

The `atlasia/darija_english` dataset lists a **CC BY-NC 4.0** license on its [Hugging Face page](https://huggingface.co/datasets/atlasia/darija_english). The data's terms and attribution requirements apply independently of this code. Corpus files are not redistributed in this repository.

[Atlaset](https://huggingface.co/datasets/atlasia/Atlaset) is another possible resource. Access requires accepting its Hugging Face terms; the current notebooks use the public `darija_english` dataset.

The notebook explanations and code were prepared for this project. The book repository is a learning reference; its notebooks and weights are not copied here.

## Planned progression

1. Prepare and inspect a Darija corpus.
2. Build the tokenizer and sliding windows.
3. Implement causal attention in PyTorch, add multi-head attention, and assemble the Transformer block.
4. Assemble a small GPT and train its parameters from scratch.
5. Evaluate generations, then study instruction adaptation.
