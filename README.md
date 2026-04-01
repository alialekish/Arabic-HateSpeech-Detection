# Arabic Hate Speech Detection.

A deep learning pipeline for detecting hate speech in Arabic tweets, comparing classical ML and neural network approaches using FastText Arabic embeddings.

---

## Overview

This project builds and evaluates multiple models to classify Arabic tweets as **hate** or **non-hate**. It covers the full pipeline: text cleaning, deduplication, embedding, model training, threshold tuning, and evaluation.

---

## Dataset

- **Levantine Arabic Hate Speech Detection Corpus**  
Available on Kaggle: [View Dataset](https://www.kaggle.com/datasets/ziedzen/levantine-arabic-hate-speech-detection-corpus)
- **Size after cleaning & deduplication:** 12,841 samples
- **Class distribution:**
  - 🔥 Hate: 7,540 (58.72%)
  - 🌱 Non-hate: 5,301 (41.28%)
- **Split:** 64% train / 16% validation / 20% test (stratified)

---

## Models & Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **FastText + LSTM** 🏆 | 81.90% | 83.32% | 86.47% | **84.87%** |
| FastText + CNN + BiLSTM | 80.38% | 82.22% | 84.95% | 83.56% |
| TF-IDF + Logistic Regression | 81.24% | 88.51% | 78.18% | 83.03% |
| FastText + BiLSTM | 79.64% | 83.12% | 81.96% | 82.54% |

> F1-score is used as the primary metric due to class imbalance.

---

## Preprocessing

- Remove URLs, mentions, hashtags
- Remove Arabic diacritics (tashkeel)
- Normalize Arabic characters (e.g., `إأآا` → `ا`)
- Remove non-Arabic characters
- Collapse repeated characters
- Remove duplicate tweets (1,022 removed — 7.37%)

---

## Architecture

- **Embeddings:** Facebook FastText Arabic (`cc.ar.300.vec`, 300-dim)
- **Vocab size:** 30,000 | **Max sequence length:** 100
- **Batch size:** 64 | **Epochs:** 8
- **Optimizer:** Adam with ReduceLROnPlateau
- **Models:** LSTM, BiLSTM, CNN+BiLSTM
- **Baseline:** TF-IDF + Logistic Regression

---

## Setup

### Requirements

```bash
pip install tensorflow scikit-learn pandas numpy
```

### FastText Embeddings

The notebook auto-downloads the FastText Arabic vectors (~4.2 GB):

```
https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.vec.gz
```

### Run

Open `ArabicHateSpeech.ipynb` in [Google Colab](https://colab.research.google.com/) with a **T4 GPU** runtime and run all cells.

---

## Repository Structure

```
├── ArabicHateSpeech.ipynb   # Full pipeline notebook
├── requirements.txt
└── README.md
```

---

## Author

**Ali Alekish**  
AI Student — Jordan University of Science and Technology  
[GitHub](https://github.com/alialekish)

---

## License

MIT License — University project built at JUST.
