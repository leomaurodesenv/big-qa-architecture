# BigQA Architecture: Big Data Architecture for Question Answering Systems

Question Answering (QA) is the task of answering questions written in natural language (typically reading comprehension questions).
QA systems can be used in a variety of use cases, such as navigate through complex knowledge bases or long documents (like a "sophisticated search engine"). A "knowledge base" can be a set of websites, an internal wiki (e.g., SharePoint or Confluence pages) or a collection of financial reports. 
This repository gathers a set of resources to build a Big Data Architecture for QA systems.

<p align="center"><img src="./docs/img/qa-example.png"></p>

*Figure source: [The Stanford Question Answering Dataset](https://rajpurkar.github.io/mlx/qa-and-squad/)*

---
## Components

Components used to instantiate the Case Study and perform the algorithm experiments.

- [Haystack](https://github.com/deepset-ai/haystack) - Open source NLP framework for Question Answering.
- [Elasticsearch](https://www.elastic.co/) - Document Store for keep the cleaned and preprocessed documents.

### Datasets

A collection of datasets used in algorithm experiments.

- [AdversarialQA](https://huggingface.co/datasets/adversarial_qa) - Complex Question Answering dataset.
- [DuoRC](https://huggingface.co/datasets/duorc) - Questions about Wikipedia and IMDb movie plots.
- [SQuAD](https://huggingface.co/datasets/squad) - Traditional QA dataset from Stanford.
