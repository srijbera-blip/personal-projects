# personal-projects
# Predicting breast cancer PAM50 subtype from gene expression

A machine learning classifier that predicts a breast tumour's PAM50 molecular
subtype (Luminal A, Luminal B, Basal-like, HER2-enriched) from its RNA-seq
gene-expression profile, using TCGA-BRCA data.



## Background

PAM50 is a 50-gene signature that classifies breast tumours into intrinsic
molecular subtypes, each with distinct prognosis and treatment implications.
This project asks whether these subtypes can be predicted from bulk RNA-seq
expression using a simple, interpretable model.

## Data

- **Source:** TCGA-BRCA cohort, downloaded from UCSC Xena
- **Expression:** STAR TPM matrix (~60,660 genes)
- **Labels:** PAM50 subtype
- After removing duplicate samples and the rare Unknown/Normal classes,
  and intersecting expression with labels: **67 samples across 4 subtypes**

## Method

1. Selected the **top 1,000 most variable genes** as features (to reduce
   overfitting given many features, few samples)
2. **Stratified** train/test split (80/20)
3. **Logistic regression** baseline classifier
4. Evaluated with a single split, **5-fold cross-validation**, and a
   confusion matrix

## Results

- Single train/test split: ~93% accuracy
- **5-fold cross-validation: ~81% mean accuracy (std ~0.08)** — the more
  reliable estimate
- Confusion matrix: near-perfect on Basal, LumA, LumB; HER2 hardest
  (smallest class)

## Limitations

- Small sample size (n = 67) and class imbalance (HER2 rarest)
- Single cohort — no external validation
- Variance-based feature selection is simple; more principled methods exist

## Running it

Requires Python with pandas, scikit-learn, matplotlib, seaborn.
Expression and label files are downloaded from UCSC Xena into `Data/`
(not included in this repo).
