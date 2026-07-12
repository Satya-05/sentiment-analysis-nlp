# Sentiment Analysis: TF-IDF vs DistilBERT

A comparison of a classical machine learning baseline against a fine-tuned transformer model for 3-class sentiment classification on Amazon product reviews.

## Project Overview

This project classifies product reviews into **negative**, **neutral**, or **positive** sentiment, comparing two fundamentally different NLP approaches:

1. **Baseline**: TF-IDF vectorization + Logistic Regression
2. **Transformer**: Fine-tuned DistilBERT (`distilbert-base-uncased`)

The goal was to quantify how much a contextual transformer model improves over a classical bag-of-words approach on a real, imbalanced, noisy dataset — and to understand *where* and *why* each approach succeeds or fails.

## Dataset

- **Source**: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab), Magazine Subscriptions category
- **Size**: 71,497 raw reviews → 65,229 after cleaning (deduplication, missing text removal)
- **Labels**: Derived from star ratings — 1-2★ → negative, 3★ → neutral, 4-5★ → positive
- **Class distribution**: Positive 72.3% / Negative 20.4% / Neutral 7.3% (significant imbalance)

**Why this dataset**: Amazon Reviews offered real, unlabeled-by-proxy text requiring a custom 3-class labeling scheme from star ratings, rather than a pre-labeled academic benchmark like IMDB. This let the project involve genuine label design and feature scoping decisions rather than consuming a dataset as-is.

## Methodology

### 1. Exploratory Data Analysis
- Identified severe class imbalance (neutral class only 7.3% of data)
- Found negative reviews are longer on average than positive reviews (median ~30 vs ~18 words)
- Discovered 11% of reviews contained raw HTML artifacts (`<br>` tags)
- Removed 8.76% duplicate reviews to prevent train/test data leakage

### 2. Preprocessing
Two parallel cleaning pipelines were built for the two modeling approaches:
- **Light cleaning** (`text_clean`): HTML stripping, contraction expansion, lowercasing — used for DistilBERT, which relies on natural sentence structure and context
- **Heavy cleaning** (`text_tfidf`): stopword removal (negation words preserved — "not", "no", "never" are critical sentiment signals), lemmatization — used for the TF-IDF baseline

### 3. Baseline Model — TF-IDF + Logistic Regression
- TF-IDF vectorization: 10,000 features, unigrams + bigrams, min_df=5, max_df=0.8
- `class_weight='balanced'` to address class imbalance (chosen over SMOTE, since synthetic TF-IDF vectors don't correspond to real sentences)
- Stratified train/test split (80/20)

### 4. Transformer Model — DistilBERT
- Fine-tuned `distilbert-base-uncased` on a stratified 12,000-row subset (CPU-only training constraint — Intel Iris Xe integrated graphics, no dedicated GPU)
- 3 epochs, batch size 8, max sequence length 128 tokens
- Best checkpoint selected by macro-F1 on validation set

**Why DistilBERT over full BERT**: DistilBERT retains ~97% of BERT's language understanding at roughly 60% of the computational cost — a meaningful difference when training on CPU-only hardware, and a realistic engineering tradeoff relevant to deployment latency as well.

**Why a 12K subset rather than the full 65K dataset**: Full fine-tuning on CPU was estimated at 5-6+ hours for 3 epochs. A stratified subset (preserving the original 72/20/7 class ratios) kept iteration feasible while still providing enough signal for a meaningful comparison against the baseline.

## Results

| Metric | Baseline (TF-IDF + LogReg) | DistilBERT | Improvement |
|---|---|---|---|
| Accuracy | 81% | 86% | +5pp |
| Negative F1 | 0.71 | 0.77 | +0.06 |
| Neutral F1 | 0.32 | 0.37 | +0.05 |
| Positive F1 | 0.91 | 0.94 | +0.03 |
| **Macro F1** | **0.65** | **0.70** | **+0.05** |

DistilBERT outperformed the baseline across every class, with the largest relative gains on neutral sentiment — though neutral remained the hardest class for both models by a wide margin.

### Key Finding: The Neutral Class Problem
Both models struggled most with neutral sentiment (F1 0.32 baseline, 0.37 DistilBERT), even though DistilBERT improved on it. This reflects genuine linguistic ambiguity in 3-star reviews rather than a limitation specific to either architecture — neutral reviews often contain a mix of positive and negative language, making them inherently harder to classify than clearly polarized text.

**Possible next steps**: more neutral training examples (only 7.3% of the dataset), or reframing the problem as ordinal regression on the 1-5 star scale rather than discrete 3-class classification.

### Known Limitation: Very Short Reviews
Testing revealed DistilBERT can misclassify very short, telegraphic reviews. For example, "Terrible. Avoid." was predicted **positive** (99.3% confidence), while the longer equivalent "This is terrible, avoid it completely." was correctly predicted **negative** (99.1% confidence). This is likely because the training data's median review length was 21 words, leaving the model under-exposed to very short inputs. The Streamlit app surfaces a reliability warning for inputs under 5 words to set correct expectations rather than hiding this limitation.

## Streamlit App

A local Streamlit app lets users enter a review and see side-by-side predictions from both models, including class probability breakdowns. Run locally with:

```powershell
streamlit run app/app.py
```

## Project Structure

```
sentiment-analysis-nlp/
├── data/                   # Raw and processed datasets (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_distilbert_model.ipynb
├── models/                 # Saved model artifacts (gitignored)
├── src/
│   └── download_data.py
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

## Tech Stack

Python, pandas, scikit-learn, NLTK, HuggingFace Transformers, PyTorch (CPU), Streamlit, BeautifulSoup
