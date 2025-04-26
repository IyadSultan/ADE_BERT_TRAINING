# Adverse Drug Reaction (ADR) Detection Models

This repository contains Jupyter notebooks for training and evaluating machine learning models to detect adverse drug reactions in clinical text.

## Contents

- [`train_BERT_on_ADR.ipynb`](./train_BERT_on_ADR.ipynb): Training pipeline for fine-tuning BERT models on ADR detection tasks
- [`evaluate_modern_adr.ipynb`](./evaluate_modern_adr.ipynb): Evaluation script for ModernBERT ADR classification model

## Overview

Adverse Drug Reactions (ADRs) are harmful or unpleasant effects caused by medications when used at normal doses. Identifying ADRs in text is crucial for patient safety and pharmacovigilance. 

This repository demonstrates two common NLP tasks for ADR detection in clinical or biomedical text:

1. **Named Entity Recognition (NER)** – extracting the specific text spans that describe ADRs in clinical narratives or reports
2. **Text Classification** – determining if a given document or sentence contains any mention of an ADR (yes/no)

## Models

### BERT-based ADR Detection

The `train_BERT_on_ADR.ipynb` notebook demonstrates how to fine-tune PubMedBERT/BiomedBERT for both classification and NER tasks using the ADE-Corpus V2 dataset. The notebook covers:

- Loading and exploring the ADE-Corpus V2 dataset
- Data preparation for both classification and NER tasks
- Model fine-tuning with appropriate evaluation metrics
- Performance analysis before and after training

### ModernBERT ADR Classification

The `evaluate_modern_adr.ipynb` notebook evaluates the performance of ModernBERT (specifically `mccoole/ModernBERT-large-ade-corpus-v2-classification`) on ADR classification. This notebook:

- Loads the pre-trained ModernBERT model
- Sets up a classification pipeline
- Tests the model on example sentences
- Includes code for comprehensive evaluation on the ADE-Corpus V2 test set

## Dataset

Both notebooks utilize the **ADE-Corpus V2** dataset, a public benchmark for adverse drug event detection consisting of sentences from biomedical reports. Each sentence is labeled whether it contains an adverse drug effect (ADE) or not, and ADR mentions are annotated within the text.

## Dependencies

- transformers
- datasets
- torch
- scikit-learn
- seqeval (for NER evaluation)
- evaluate
- numpy

## Usage

1. Open the notebooks in a Jupyter environment with Python 3.x
2. Ensure all dependencies are installed
3. Run the cells sequentially

For training, a GPU environment is recommended (the notebooks are configured for Google Colab).

## Results

The models achieve significant improvements after fine-tuning:

| Task | Metric | Before Training | After Fine-tuning |
|------|--------|-----------------|-------------------|
| Classification | F1 Score | ~45% | ~93% |
| NER | F1 Score | ~0% | ~91% |

## Citations

The ADR detection models are trained on the ADE-Corpus V2 dataset:

```
@article{gurulingappa2012development,
  title={Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports},
  author={Gurulingappa, Harsha and Rajput, Abdul Mateen and Roberts, Angus and Fluck, Juliane and Hofmann-Apitius, Martin and Toldo, Luca},
  journal={Journal of biomedical informatics},
  volume={45},
  number={5},
  pages={885--892},
  year={2012},
  publisher={Elsevier}
}
```

## License

Please refer to the licenses of the respective models and datasets used in this project.
