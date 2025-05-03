# BigQA Python Scripts

This folder contains Python scripts and utilities for the BigQA architecture. The scripts are designed to support various tasks related to question-answering systems, including data preprocessing, model fine-tuning, and evaluation.

The `python` folder serves as implementation of heavy workloads; it includes:

- **Model Fine-Tuning**: Code for fine-tuning document reader models on specific datasets.
- **Experiments**: Tools for running experiments on document retrievers and readers.

---
## Model Fine-Tuning

### Setup Environment

```sh
# Creating a virtual environment
$ python -m venv .venv
$ source .venv/bin/activate
# Installing packages
$ pip install -r python/requirements.txt
```

### Hugging Face Credentials

```sh
# Create .env file
$ touch .env
# Write credentials
$ echo "HUGGINGFACE_TOKEN=\"hf_xxxxxxxxxxxxxxxxxxxx\"" >> .env
```
### Training the Document Reader

```sh
# Verify the available arguments
$ python /python/document_reader_finetuning.py --help
# Fine-tune the model, and publish on HuggingFace
$ python /python/document_reader_finetuning.py --doc_reader "BERT" --sport "BASKETBALL"
$ python /python/document_reader_finetuning.py --doc_reader "RoBERTa" --sport "SOCCER"
```
